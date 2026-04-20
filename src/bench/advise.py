"""Ship the practitioner rule as a CLI.

    # Ask the advisor what to deploy on T4 + Llama-3.2 1B/3B:
    python -m src.bench.advise --hw T4 --draft 1B --target 3B

    # Or with a custom hardware spec:
    python -m src.bench.advise --bandwidth-gbps 400 --fixed-overhead-ms 8 \\
        --draft 7B --target 70B --alpha-k3 0.82
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.specdec.advisor import (
    A100_PYTORCH_EAGER,
    H100_PYTORCH_EAGER,
    HardwareSpec,
    T4_PYTORCH_EAGER,
    recommend,
)

PRESETS = {
    "T4": T4_PYTORCH_EAGER,
    "A100": A100_PYTORCH_EAGER,
    "H100": H100_PYTORCH_EAGER,
}

SIZE_SHORTHANDS = {
    "68M": 6.8e7, "200M": 2e8, "500M": 5e8,
    "1B": 1e9, "3B": 3e9, "7B": 7e9, "8B": 8e9,
    "13B": 1.3e10, "34B": 3.4e10, "70B": 7e10,
}


def parse_size(s: str) -> float:
    s = s.upper().strip()
    if s in SIZE_SHORTHANDS:
        return SIZE_SHORTHANDS[s]
    return float(s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hw", choices=list(PRESETS), help="Hardware preset.")
    p.add_argument("--bandwidth-gbps", type=float,
                   help="HBM bandwidth. Overrides preset if given.")
    p.add_argument("--fixed-overhead-ms", type=float,
                   help="Per-call kernel-launch+layernorm+RoPE overhead.")
    p.add_argument("--draft", required=True,
                   help="Draft-model parameter count (e.g. 1B, 500M).")
    p.add_argument("--target", required=True,
                   help="Target-model parameter count.")
    p.add_argument("--alpha-k3", type=float, default=None,
                   help="Expected chain acceptance at k=3 (optional).")
    p.add_argument("--prompt-overlap", action="store_true",
                   help="Workload has prompt/output overlap (code, RAG, summarisation).")
    args = p.parse_args()

    if args.hw:
        base = PRESETS[args.hw]
        hw = HardwareSpec(
            name=base.name,
            bandwidth_gbps=args.bandwidth_gbps or base.bandwidth_gbps,
            fixed_overhead_ms=(args.fixed_overhead_ms
                               if args.fixed_overhead_ms is not None
                               else base.fixed_overhead_ms),
        )
    else:
        if args.bandwidth_gbps is None or args.fixed_overhead_ms is None:
            p.error("Either --hw or (--bandwidth-gbps AND --fixed-overhead-ms) "
                    "must be given.")
        hw = HardwareSpec(name="custom",
                          bandwidth_gbps=args.bandwidth_gbps,
                          fixed_overhead_ms=args.fixed_overhead_ms)

    rec = recommend(
        hw=hw,
        params_draft=parse_size(args.draft),
        params_target=parse_size(args.target),
        expected_alpha_k3=args.alpha_k3,
        workload_has_prompt_overlap=args.prompt_overlap,
    )

    print(f"Hardware:          {hw.name} "
          f"({hw.bandwidth_gbps:.0f} GB/s, Tf={hw.fixed_overhead_ms} ms)")
    print(f"Model pair:        {args.draft} -> {args.target}")
    print(f"Predicted c:       {rec.predicted_c:.3f}")
    print(f"Break-even alpha:  {rec.breakeven_alpha_k3:.3f}  (for k=3)")
    print(f"Recommendation:    {rec.family}")
    print(f"Reason:            {rec.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
