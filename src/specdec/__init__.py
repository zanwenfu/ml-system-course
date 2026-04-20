"""Speculative decoding measurement library (Duke CS 590, Spring 2026).

Small, intentionally minimal: one module per measurement subsystem named
in the paper. Notebooks and CLI entry points are thin wrappers over these
primitives.
"""

from .advisor import (
    HardwareSpec,
    Recommendation,
    recommend,
    T4_PYTORCH_EAGER,
    A100_PYTORCH_EAGER,
    H100_PYTORCH_EAGER,
)
from .cost_model import (
    closed_form_c,
    linear_sd_breakeven_alpha,
    linear_sd_predicted_speedup,
    sequoia_dp,
    sequoia_tree_speedup,
)
from .probe import per_call_probe_summary
from .profiler_attr import categorize_events

__all__ = [
    "HardwareSpec",
    "Recommendation",
    "recommend",
    "T4_PYTORCH_EAGER",
    "A100_PYTORCH_EAGER",
    "H100_PYTORCH_EAGER",
    "closed_form_c",
    "linear_sd_breakeven_alpha",
    "linear_sd_predicted_speedup",
    "sequoia_dp",
    "sequoia_tree_speedup",
    "per_call_probe_summary",
    "categorize_events",
]
