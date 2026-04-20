"""Tests that the persisted per-call probe satisfies the paper's claim.

If a future re-run adds or changes cache lengths, these tests pin the
bandwidth-floor finding so regressions (or JSON-schema drift) are caught.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.specdec.probe import per_call_probe_summary


def test_cd_flat_across_cache_lengths():
    summary = per_call_probe_summary(
        ROOT / "src" / "results" / "cd_probe_results.json")
    # README §8 claim: Cd is flat at 20.0 +/- 0.2 ms, 2.3% spread.
    assert 19.5 <= summary["Cd_mean_ms"] <= 20.5
    assert summary["Cd_relative_spread"] < 0.05


def test_tree_runtime_results_pin_the_paper_numbers():
    import json
    v2 = json.loads(
        (ROOT / "src" / "results" / "phase2b_tree_results.json").read_text())
    v3 = json.loads(
        (ROOT / "src" / "results" / "phase2b_v3_results.json").read_text())

    v2_n16 = next(s for s in v2["summaries"] if s["method"] == "tree_n16_d3")
    v3_n16 = next(s for s in v3["summaries"] if s["method"] == "tree_n16_d3")

    # README headline: 1.68x predicted, 0.56x v2 measured, 0.46x v3 measured.
    assert abs(v2_n16["predicted_speedup"] - 1.68) < 0.01
    assert abs(v2_n16["speedup_vs_baseline"] - 0.557) < 0.01
    assert abs(v3_n16["speedup_vs_baseline"] - 0.456) < 0.01

    # The fourth-assumption finding: v3 is measurably slower than v2.
    assert v3_n16["speedup_vs_baseline"] < v2_n16["speedup_vs_baseline"]
