"""Tests for the deployable advisor.

These pin the paper's practitioner rule as behavioural assertions.
Regressions here would mean the advisor has drifted from the rule
stated in README §Practitioner rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.specdec.advisor import (
    A100_PYTORCH_EAGER,
    H100_PYTORCH_EAGER,
    T4_PYTORCH_EAGER,
    recommend,
)


def test_t4_plus_llama32_is_the_papers_regime():
    # The paper: T4 + 1B/3B -> PLD only.
    rec = recommend(T4_PYTORCH_EAGER, params_draft=1e9, params_target=3e9,
                    expected_alpha_k3=0.732)
    assert rec.family == "pld"
    assert 0.55 <= rec.predicted_c <= 0.65
    # Break-even on T4 at k=3 is 0.697 in the paper.
    assert abs(rec.breakeven_alpha_k3 - 0.697) < 0.02


def test_a100_sequoia_setup_recommends_tree():
    # Sequoia's setup: A100 + 68M/7B with high alpha -> tree wins.
    rec = recommend(A100_PYTORCH_EAGER,
                    params_draft=6.8e7, params_target=7e9,
                    expected_alpha_k3=0.80)
    assert rec.family == "tree"
    # Compute-bound regime: c should approach params ratio ~0.01.
    assert rec.predicted_c < 0.10


def test_prompt_overlap_always_picks_pld():
    # Code completion / RAG: PLD dominates regardless of regime.
    for hw in (T4_PYTORCH_EAGER, A100_PYTORCH_EAGER, H100_PYTORCH_EAGER):
        rec = recommend(hw, params_draft=1e9, params_target=70e9,
                        workload_has_prompt_overlap=True)
        assert rec.family == "pld"


def test_compute_bound_low_alpha_says_dont_speculate():
    # If alpha is below break-even, advisor must decline SD entirely.
    rec = recommend(A100_PYTORCH_EAGER,
                    params_draft=5e8, params_target=7e9,
                    expected_alpha_k3=0.20)
    assert rec.family == "dont_speculate"


def test_recommendation_is_stable_across_identical_inputs():
    r1 = recommend(T4_PYTORCH_EAGER, 1e9, 3e9, expected_alpha_k3=0.7)
    r2 = recommend(T4_PYTORCH_EAGER, 1e9, 3e9, expected_alpha_k3=0.7)
    assert r1 == r2
