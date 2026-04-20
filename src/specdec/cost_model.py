"""Closed-form cost models for speculative decoding.

All quantities are wall-clock milliseconds unless noted. No PyTorch /
CUDA dependency: pure arithmetic, safe to unit-test and pin.

Reference: Leviathan et al. 2023 (linear SD), Chen et al. 2024 (Sequoia
tree SD). The per-call decomposition `C = Tw + Tf` is this project's
contribution and is the operational version of the paper's §12.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence


def closed_form_c(
    params_draft: float,
    params_target: float,
    bandwidth_gbps: float,
    fixed_overhead_ms: float,
    bytes_per_param: int = 2,
) -> float:
    """Predict the per-call cost ratio c = Cd/Ct from first principles.

    C = Tw + Tf where Tw = params * bytes / bandwidth is weight-loading
    time and Tf is per-call fixed overhead (kernel launch, layernorm,
    RoPE, residuals). Two regimes:

      * Tf << Tw: c -> params_d / params_t  (compute-bound; Sequoia-style
        assumptions hold).
      * Tf ~= Tw: c -> 1                    (bandwidth-bound; speculative
        decoding breaks).

    For T4 + Llama-3.2 1B/3B with bandwidth=300 GB/s and Tf=13 ms this
    predicts c ~= 0.60; measured c is 0.596.
    """
    tw_d = (params_draft * bytes_per_param) / (bandwidth_gbps * 1e9) * 1e3
    tw_t = (params_target * bytes_per_param) / (bandwidth_gbps * 1e9) * 1e3
    return (tw_d + fixed_overhead_ms) / (tw_t + fixed_overhead_ms)


def linear_sd_breakeven_alpha(k: int, c: float) -> float:
    """Minimum acceptance rate for linear SD at k draft tokens to break even.

    From Leviathan et al. 2023: speedup > 1 iff alpha > (k*c + 1) / (k + 1).
    On T4 with c ~= 0.6 and k=3 the threshold is 0.70 -- measured alpha is
    0.73, which is why k=3 clears the break-even by only 3 percentage points
    and larger k slips into slowdown.
    """
    return (k * c + 1.0) / (k + 1.0)


def linear_sd_predicted_speedup(alpha: float, k: int, c: float) -> float:
    """Leviathan et al. 2023 Eq. 3 (greedy, geometric-length)."""
    if alpha == 1.0:
        return (k + 1) / (k * c + 1.0)
    numerator = 1.0 - alpha ** (k + 1)
    denominator = (1.0 - alpha) * (k * c + 1.0)
    return numerator / denominator


def sequoia_tree_speedup(
    tree_size: int,
    depth: int,
    acceptance_vector: Sequence[float],
    c: float,
    t_n_overhead: float = 1.0,
) -> tuple[float, float]:
    """Sequoia tree-SD speedup given a (tree_size, depth) shape.

    Returns (speedup_ratio, expected_accepted_tokens_G). `t_n_overhead`
    is Sequoia's tree-verify inflation factor t(n); their paper assumes
    t(16) ~= 1.15 on A100. On T4 we measured t(16) = 1.83 -- the second
    hidden assumption from the README.
    """
    g = _expected_accepted_tokens(tree_size, depth, acceptance_vector)
    iter_cost = depth * c * 1.0 + t_n_overhead  # units: Ct
    baseline_iter_cost = g * 1.0  # g baseline forwards
    return baseline_iter_cost / iter_cost, g


def _expected_accepted_tokens(n: int, d: int, p: Sequence[float]) -> float:
    """Closed-form E[accepted tokens] for a (n, d) balanced tree.

    For a balanced tree with depth d and n nodes, each position i in
    [0, d) accepts with probability p[i] given its ancestors accepted.
    G = sum_i prod_{j<=i} p[j].
    """
    total = 0.0
    running = 1.0
    for i in range(d):
        if i >= len(p):
            break
        running *= p[i]
        total += running
    return total


def sequoia_dp(
    max_nodes: int,
    max_depth: int,
    acceptance_vector: Sequence[float],
    c: float,
) -> dict[tuple[int, int], float]:
    """Port of Sequoia's DP tree optimizer (Chen et al. 2024, Alg. 1).

    Computes the DP-optimal tree shape under the budget (max_nodes,
    max_depth) against a measured positional acceptance vector. Returns
    a dict keyed on (nodes, depth) -> expected accepted tokens. The
    caller picks the argmax and feeds it into `sequoia_tree_speedup`.

    This is a faithful port, not an extension: intentionally matches
    the paper's formulation so the 3x gap in the README can be attributed
    to hidden assumptions in the cost model, not to a DP bug.
    """
    p = tuple(acceptance_vector)

    @lru_cache(maxsize=None)
    def best(nodes: int, depth: int) -> float:
        if nodes <= 0 or depth <= 0:
            return 0.0
        if depth == 1:
            return p[0] if nodes >= 1 else 0.0
        # Try every split of remaining budget across children.
        best_val = 0.0
        for first_child in range(1, nodes + 1):
            child_val = best(first_child, depth - 1)
            rest = best(nodes - first_child, depth)
            total = p[0] + p[0] * child_val + rest
            if total > best_val:
                best_val = total
        return best_val

    return {(n, d): best(n, d) for n in range(1, max_nodes + 1)
            for d in range(1, max_depth + 1)}
