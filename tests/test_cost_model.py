"""Unit tests for the cost-model primitives.

These tests encode the paper's headline quantitative claims so that
regressions in the library show up as test failures.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.specdec.cost_model import (
    closed_form_c,
    linear_sd_breakeven_alpha,
    linear_sd_predicted_speedup,
    sequoia_dp,
    sequoia_tree_speedup,
)


def test_closed_form_c_matches_t4_measurement():
    # T4 HBM ~300 GB/s, Llama-3.2 1B/3B, Tf ~= 13 ms.
    # README §'A closed-form heuristic' predicts c ~= 0.60; measured 0.596.
    c = closed_form_c(
        params_draft=1_000_000_000,
        params_target=3_000_000_000,
        bandwidth_gbps=300,
        fixed_overhead_ms=13.0,
    )
    assert 0.58 <= c <= 0.62, f"closed-form c={c:.3f} outside [0.58, 0.62]"


def test_closed_form_c_compute_bound_limit():
    # Tf << Tw -> c -> params ratio. Set Tf tiny.
    c = closed_form_c(1_000_000_000, 3_000_000_000,
                      bandwidth_gbps=300, fixed_overhead_ms=0.001)
    assert abs(c - 1 / 3) < 0.01


def test_linear_sd_breakeven_k3_on_t4():
    # README Table 1: alpha_measured=0.732 just clears breakeven 0.697.
    alpha_star = linear_sd_breakeven_alpha(k=3, c=0.596)
    assert abs(alpha_star - 0.697) < 0.005


def test_linear_sd_predicted_speedup_bounds():
    # At alpha=1 the predicted speedup equals (k+1)/(kc+1).
    for k in (1, 3, 5, 10):
        pred = linear_sd_predicted_speedup(alpha=1.0, k=k, c=0.596)
        expected = (k + 1) / (k * 0.596 + 1)
        assert math.isclose(pred, expected, rel_tol=1e-6)


def test_sequoia_dp_returns_strictly_positive_values():
    p = [0.846, 0.575, 0.55, 0.333, 0.167]
    table = sequoia_dp(max_nodes=8, max_depth=3, acceptance_vector=p, c=0.596)
    # DP must be monotonic in budget.
    assert table[(8, 3)] >= table[(4, 3)] >= table[(1, 3)]
    assert table[(8, 3)] >= table[(8, 1)]


def test_sequoia_tree_speedup_returns_sensible_bounds():
    # The balanced-tree helper is a sanity bound, not a DP replacement;
    # the full Sequoia DP lives in the saved JSON results and is
    # exercised by test_probe.py::test_tree_runtime_results_pin_the_paper_numbers.
    # Here we only check arithmetic invariants.
    p = [0.846, 0.575, 0.55, 0.333, 0.167]
    speedup, g = sequoia_tree_speedup(
        tree_size=16, depth=3, acceptance_vector=p,
        c=0.333, t_n_overhead=1.15,
    )
    assert speedup > 0 and g > 0
    assert g <= 3.0  # at most 3 accepted tokens for depth=3
    # E[G] = 0.846 + 0.846*0.575 + 0.846*0.575*0.55 ~= 1.64.
    assert abs(g - 1.64) < 0.05
