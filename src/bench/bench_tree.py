"""Reproduce the Sequoia DP optimizer + tree-SD cost decomposition (README §9).

Two modes:

  * --from-results (default): loads saved tree measurements and prints
    the calibrated-vs-uncalibrated speedup. No GPU needed. This is what
    lets a reviewer verify the paper's central 1.68x -> 0.56x gap and
    the 1.1% reconciliation without running the full pipeline.
  * --cuda: run v2 or v3 tree runtime on a live model. Delegates to
    src/notebooks/phase2b_{with_cd_probe,v3_persistent_cache}.ipynb.

Example
-------
    # Reconcile Sequoia's 1.68x prediction with the measured 0.56x:
    python -m src.bench.bench_tree --tree-config 16x3 --version v2

    # Fourth-assumption finding (v3 should be slower, not faster):
    python -m src.bench.bench_tree --tree-config 16x3 --version v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.specdec.cost_model import sequoia_dp, sequoia_tree_speedup

RESULTS = ROOT / "src" / "results"


def load_acceptance_vector() -> list[float]:
    data = json.loads((RESULTS / "swor_acceptance_vector.json").read_text())
    return data.get("acceptance_vector") or data["acceptance"]


def summarize_tree_result(version: str, n: int, d: int) -> None:
    path = RESULTS / (
        "phase2b_tree_results.json" if version == "v2"
        else "phase2b_v3_results.json"
    )
    data = json.loads(path.read_text())

    method = f"tree_n{n}_d{d}"
    summary = next(
        (s for s in data["summaries"] if s["method"] == method), None)
    if summary is None:
        print(f"No saved summary for {method} in {path.name}.", file=sys.stderr)
        print(f"Available: {[s['method'] for s in data['summaries']]}",
              file=sys.stderr)
        return

    profiler = data.get("profiler_snapshots", {}).get(f"n{n}_d{d}")
    baseline_tps = data["baseline_tps"]

    print(f"=== Tree SD {version}, (n={n}, d={d}) ===")
    print(f"Sequoia prediction:    {summary['predicted_speedup']:.3f}x")
    print(f"Measured on T4:        {summary['speedup_vs_baseline']:.3f}x "
          f"({summary['mean_tok_sec']:.2f} tok/s vs baseline "
          f"{baseline_tps:.2f} tok/s)")
    print(f"Trials:                {summary['n_trials']} "
          f"(std {summary['std_tok_sec']:.2f} tok/s)")
    print(f"Tokens / iter (meas):  {summary['mean_tokens_per_iter']:.2f}")

    if profiler:
        print("\nPer-iteration decomposition (ms):")
        keys = [
            ("draft_prefix_ms_per_iter",  "draft prefix rebuild"),
            ("draft_extend_ms_per_iter",  "draft extends"),
            ("verify_ms_per_iter",        "target verify"),
            ("cache_extend_ms_per_iter",  "cache extend (v3 only)"),
        ]
        total = 0.0
        for k, label in keys:
            v = profiler.get(k)
            if v is None:
                continue
            print(f"  {label:<25s} {v:>8.2f}")
            total += v
        print(f"  {'iter total':<25s} {total:>8.2f}")


def show_dp(max_nodes: int, max_depth: int, c: float) -> None:
    p_vec = load_acceptance_vector()
    table = sequoia_dp(max_nodes, max_depth, p_vec, c)
    # Print the DP-optimal shape and its predicted G.
    best_shape = max(table, key=table.get)
    print(f"Sequoia DP-optimal shape at budget ({max_nodes}, {max_depth}): "
          f"n={best_shape[0]}, d={best_shape[1]}, E[G]={table[best_shape]:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tree-config", default="16x3",
                   help="<n>x<d>. Examples: 8x3, 16x3.")
    p.add_argument("--version", choices=("v2", "v3"), default="v2")
    p.add_argument("--from-results", action="store_true", default=True)
    p.add_argument("--cuda", action="store_true",
                   help="Delegate to notebook path (T4 required).")
    p.add_argument("--show-dp", action="store_true",
                   help="Print DP-optimal shape at the given budget.")
    p.add_argument("--c", type=float, default=0.596,
                   help="Measured Cd/Ct. Sequoia's A100 value is ~0.333.")
    args = p.parse_args()

    n_str, d_str = args.tree_config.lower().split("x")
    n, d = int(n_str), int(d_str)

    if args.cuda:
        nb = ("phase2b_with_cd_probe.ipynb" if args.version == "v2"
              else "phase2b_v3_persistent_cache.ipynb")
        print(f"Full CUDA path: src/notebooks/{nb}. Open in Colab with a T4.",
              file=sys.stderr)
        return 2

    if args.show_dp:
        show_dp(n, d, args.c)
        print()
    summarize_tree_result(args.version, n, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
