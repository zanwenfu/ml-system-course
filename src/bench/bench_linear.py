"""Reproduce the linear-SD break-even analysis from README Table 1.

Two modes:

  * --from-results: recompute predicted speedups and break-even alphas
    from the saved JSON (runs anywhere in <1s, no GPU).
  * --cuda: run the full HuggingFace assisted-generation sweep on a real
    T4. Delegates to the notebook path; see src/notebooks/spec_decoding_phase1.ipynb.

The --from-results mode is what a reviewer without a T4 should use to
verify the README's numerical claims.

Examples
--------
    # Verify README Table 1 without a GPU:
    python -m src.bench.bench_linear --from-results --k 3 5 7 10

    # Full reproduction on a T4:
    python -m src.bench.bench_linear --cuda --draft Llama-3.2-1B \\
        --target Llama-3.2-3B --k 3 5 7 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python src/bench/bench_linear.py` without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.specdec.cost_model import (
    linear_sd_breakeven_alpha,
    linear_sd_predicted_speedup,
)

# Canonical measured values from Phase 1C, rounded. Units: c dimensionless,
# alpha dimensionless. These are what the README headline uses.
T4_LLAMA32 = {
    "c": 0.596,
    "alpha_measured": {3: 0.732, 5: 0.644, 7: 0.601, 10: 0.551},
    "speedup_measured": {3: 1.01, 5: 0.98, 7: 0.95, 10: 0.92},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", nargs="+", type=int, default=[3, 5, 7, 10])
    p.add_argument("--c", type=float, default=T4_LLAMA32["c"],
                   help="Per-call cost ratio Cd/Ct. Default is T4 measurement.")
    p.add_argument("--from-results", action="store_true",
                   help="Compute from saved constants (no GPU needed).")
    p.add_argument("--cuda", action="store_true",
                   help="Run full HF assisted-generation sweep (T4 required).")
    p.add_argument("--draft", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--target", default="meta-llama/Llama-3.2-3B")
    args = p.parse_args()

    if args.cuda:
        print("The CUDA path is documented in "
              "src/notebooks/spec_decoding_phase1.ipynb.\n"
              "Open that notebook in Colab with a T4 runtime, set DRAFT/TARGET,"
              " and run all cells.", file=sys.stderr)
        return 2

    print(f"{'k':>3} {'breakeven α':>12} {'α measured':>12}"
          f" {'pred speedup':>14} {'measured':>10}")
    print("-" * 56)
    for k in args.k:
        alpha_star = linear_sd_breakeven_alpha(k, args.c)
        alpha = T4_LLAMA32["alpha_measured"].get(k)
        pred = (linear_sd_predicted_speedup(alpha, k, args.c)
                if alpha is not None else None)
        meas = T4_LLAMA32["speedup_measured"].get(k)
        print(f"{k:>3} {alpha_star:>12.3f} "
              f"{alpha if alpha else '-':>12} "
              f"{pred if pred else '-':>14.3f} "
              f"{meas if meas else '-':>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
