"""Deployable advisor: encodes the README's practitioner rule as an API.

Given a hardware spec and a (draft, target) model pair, returns the
recommended speculative-decoding family plus a one-line rationale. This
is the operational version of the paper's finding: you don't need to
run Sequoia's DP; you need to check whether bandwidth or compute is the
binding constraint, and pick accordingly.

    >>> from src.specdec.advisor import recommend, HardwareSpec
    >>> rec = recommend(
    ...     hw=HardwareSpec("T4", bandwidth_gbps=300, fixed_overhead_ms=13),
    ...     params_draft=1_000_000_000,
    ...     params_target=3_000_000_000,
    ... )
    >>> rec.family
    'pld'
    >>> rec.reason
    'c=0.60 (bandwidth-bound regime); tree methods won't break even.'
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .cost_model import closed_form_c, linear_sd_breakeven_alpha

Family = Literal["pld", "linear_k3", "tree", "dont_speculate"]


@dataclass(frozen=True)
class HardwareSpec:
    """Minimum information needed to predict the regime.

    `fixed_overhead_ms` is per-call kernel-launch + layernorm + RoPE +
    residual overhead. Measure it once per (framework, hardware) with
    the per-call probe (src/bench/probe_percall.py); rule of thumb:
      * PyTorch eager on T4: ~13 ms
      * PyTorch eager on A100: ~0.4 ms
      * CUDA-graph-fused (vLLM): ~0 ms to ~2 ms depending on backend.
    """
    name: str
    bandwidth_gbps: float
    fixed_overhead_ms: float


@dataclass(frozen=True)
class Recommendation:
    family: Family
    predicted_c: float
    breakeven_alpha_k3: float
    reason: str


# Regime threshold: c above this means bandwidth-bound (tree methods
# will underperform their Sequoia-style predictions by >2x on T4-class
# hardware). Chosen to match the measured T4 boundary (c=0.596) and
# flagged as the main knob a practitioner would tune.
C_BANDWIDTH_BOUND_THRESHOLD = 0.40


def recommend(
    hw: HardwareSpec,
    params_draft: float,
    params_target: float,
    expected_alpha_k3: float | None = None,
    workload_has_prompt_overlap: bool = False,
) -> Recommendation:
    """Pick a speculative-decoding family for the given deployment.

    Rule (paper §Practitioner rule):
      1. Compute c = Cd/Ct from the closed form.
      2. Compute break-even alpha for k=3.
      3. If c > threshold: we're bandwidth-bound. PLD only; tree and linear
         lose.
      4. Else compute-bound: linear k=3 if expected_alpha > breakeven,
         otherwise tree.
      5. If workload has prompt-output overlap (code completion, RAG,
         summarisation): PLD is always strictly better, regardless of regime.
    """
    c = closed_form_c(params_draft, params_target,
                      bandwidth_gbps=hw.bandwidth_gbps,
                      fixed_overhead_ms=hw.fixed_overhead_ms)
    alpha_star = linear_sd_breakeven_alpha(k=3, c=c)

    if workload_has_prompt_overlap:
        return Recommendation(
            family="pld", predicted_c=c, breakeven_alpha_k3=alpha_star,
            reason=f"c={c:.2f}; prompt-output overlap makes PLD "
                   f"strictly dominant (Cr~0 bypasses the per-call floor).",
        )

    if c > C_BANDWIDTH_BOUND_THRESHOLD:
        return Recommendation(
            family="pld", predicted_c=c, breakeven_alpha_k3=alpha_star,
            reason=f"c={c:.2f} (bandwidth-bound regime); "
                   f"tree methods won't break even.",
        )

    if expected_alpha_k3 is not None and expected_alpha_k3 <= alpha_star:
        return Recommendation(
            family="dont_speculate", predicted_c=c,
            breakeven_alpha_k3=alpha_star,
            reason=f"c={c:.2f} and expected alpha={expected_alpha_k3:.2f} "
                   f"<= breakeven {alpha_star:.2f}. Plain AR is faster.",
        )

    if expected_alpha_k3 is not None and expected_alpha_k3 > 0.75:
        return Recommendation(
            family="tree", predicted_c=c, breakeven_alpha_k3=alpha_star,
            reason=f"c={c:.2f} (compute-bound) and high alpha={expected_alpha_k3:.2f}; "
                   f"Sequoia-style tree SD pays off.",
        )

    return Recommendation(
        family="linear_k3", predicted_c=c, breakeven_alpha_k3=alpha_star,
        reason=f"c={c:.2f} (compute-bound). Linear SD at k=3 is the "
               f"safe default; verify alpha > {alpha_star:.2f} in production.",
    )


# Curated hardware presets (empirical fixed-overhead estimates).
T4_PYTORCH_EAGER = HardwareSpec("T4", bandwidth_gbps=300, fixed_overhead_ms=13.0)
A100_PYTORCH_EAGER = HardwareSpec("A100", bandwidth_gbps=1555, fixed_overhead_ms=0.4)
H100_PYTORCH_EAGER = HardwareSpec("H100", bandwidth_gbps=3350, fixed_overhead_ms=0.3)
