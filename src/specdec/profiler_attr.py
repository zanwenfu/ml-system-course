"""PyTorch-profiler event categorisation (Phase 3).

Pulled out of the notebooks so it can be unit-tested and reused. The
kernel-name heuristics are the same ones used in the paper; they are
threshold-based and will silently mis-attribute if CUTLASS/cuBLAS ship
a renamed kernel, which is why the probe (`specdec.probe`) is the
ground truth rather than the profiler.
"""
from __future__ import annotations

from typing import Iterable, Mapping

MATMUL_KEYS = ("gemm", "gemv", "cutlass", "ampere_", "sm80_", "mm_", "bmm")
ATTENTION_KEYS = ("flash_attn", "sdpa", "attention", "softmax")
MEMORY_KEYS = ("memcpy", "memset", "copy_", "cat", "view", "stack")
NORM_KEYS = ("layer_norm", "rms_norm", "norm_")
ELEMENTWISE_KEYS = ("elementwise", "fused", "_kernel", "add_", "mul_", "silu",
                    "gelu", "rotary", "rope")


def _bucket(name: str) -> str:
    n = name.lower()
    for key_set, bucket in (
        (MATMUL_KEYS, "matmul"),
        (ATTENTION_KEYS, "attention"),
        (MEMORY_KEYS, "memory"),
        (NORM_KEYS, "norm"),
        (ELEMENTWISE_KEYS, "elementwise"),
    ):
        if any(k in n for k in key_set):
            return bucket
    return "other"


def categorize_events(events: Iterable[Mapping]) -> dict[str, float]:
    """Sum CUDA time (us) per category from a list of profiler events.

    Each event must expose `name` and `cuda_time_us` (or `device_time`).
    Returns absolute microseconds, not percentages -- taking percentages
    earlier is the single most common way to misread a profiler when
    different methods have different total times.
    """
    totals = {
        "matmul": 0.0, "attention": 0.0, "memory": 0.0,
        "norm": 0.0, "elementwise": 0.0, "other": 0.0,
    }
    for ev in events:
        name = ev.get("name") or ev.get("key") or ""
        us = ev.get("cuda_time_us")
        if us is None:
            us = ev.get("device_time", 0.0)
        totals[_bucket(name)] += float(us)
    totals["total"] = sum(totals.values())
    return totals
