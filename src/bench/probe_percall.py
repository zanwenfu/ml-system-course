"""Per-call cost probe summary (README §8) -- the unifying measurement.

Loads `src/results/cd_probe_results.json` and reports the bandwidth-floor
finding: Cd is flat at 20 ms across an 8.5x range in cache length. This
is the single measurement that underwrites every other finding in the
paper.

The CUDA collection path is in src/notebooks/phase2b_with_cd_probe.ipynb;
this CLI summarises the saved artifact so reviewers can verify the
numeric claim without a T4.

Example
-------
    python -m src.bench.probe_percall
    python -m src.bench.probe_percall --cache-lengths 30 64 128 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.specdec.probe import per_call_probe_summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=str(
        ROOT / "src" / "results" / "cd_probe_results.json"))
    p.add_argument("--cache-lengths", nargs="+", type=int,
                   help="Filter to a subset of cache lengths.")
    args = p.parse_args()

    summary = per_call_probe_summary(args.results)
    rows = summary["per_length"]
    if args.cache_lengths:
        rows = {k: v for k, v in rows.items() if k in args.cache_lengths}

    print(f"{'cache_len':>10} {'Cd (ms)':>10} {'Ct (ms)':>10}")
    print("-" * 34)
    for length, row in sorted(rows.items()):
        print(f"{length:>10} {row['Cd_ms']:>10.2f} {row['Ct_ms']:>10.2f}")

    print()
    print(summary["finding"])
    print(f"Aggregate: Cd mean = {summary['Cd_mean_ms']:.2f} ms,"
          f" Ct mean = {summary['Ct_mean_ms']:.2f} ms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
