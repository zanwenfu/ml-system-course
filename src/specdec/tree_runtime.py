"""Minimal Sequoia-compliant tree speculative-decoding runtime.

Two variants, matching the paper:

  * v2 (within-iter KV): prefix rebuilt each iteration, tree attention
    with cover property, branch-level fork / consume-in-place semantics.
  * v3 (cross-iter KV): both draft and target KV persist across
    iterations; eliminates prefix rebuild but adds a new cache-extension
    cost per iteration. This is the optimisation whose sign flips on
    bandwidth-bound hardware.

This module deliberately avoids CUDA graphs, fused kernels, and batch
parallelism -- the paper's claim is about the *algorithmic cost model*,
and adding systems engineering would conflate the two.

Both paths pass the same correctness gate: 20/20 token-identical output
against greedy autoregressive baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TreeNode:
    """Node in the tree of speculated tokens.

    Children are indexed sequentially; `parent_idx` points back at the
    flat index used in the tree-attention mask, so ancestors are just
    `parent_idx` chain until -1.
    """
    token_id: int
    logprob: float
    parent_idx: int = -1
    depth: int = 0
    children: list["TreeNode"] = field(default_factory=list)


def build_tree_attention_mask(nodes: list[TreeNode]) -> list[list[bool]]:
    """Ancestor-only attention mask in O(n * depth).

    Row i can attend to column j iff j is an ancestor of i (or j == i).
    Used verbatim by both v2 and v3 verify paths.
    """
    n = len(nodes)
    mask = [[False] * n for _ in range(n)]
    for i, node in enumerate(nodes):
        mask[i][i] = True
        cur = node.parent_idx
        while cur != -1:
            mask[i][cur] = True
            cur = nodes[cur].parent_idx
    return mask


def accept_longest_path(nodes: list[TreeNode], target_argmax: list[int]) -> list[int]:
    """Greedy acceptance: descend while draft token == target argmax.

    Sequoia's correctness property: the accepted path length equals the
    number of verify-step positions at which draft matched target in
    greedy mode. Returns the ordered list of accepted node indices.
    """
    accepted: list[int] = []
    # Root is always accepted (it's the prompt-conditioned prefix).
    candidates = [i for i, n in enumerate(nodes) if n.parent_idx == -1]
    while candidates:
        matched = None
        for idx in candidates:
            if nodes[idx].token_id == target_argmax[idx]:
                matched = idx
                break
        if matched is None:
            break
        accepted.append(matched)
        candidates = [c for c, n in enumerate(nodes) if n.parent_idx == matched]
    return accepted


@dataclass
class IterationStats:
    """Per-iteration timing. All units are milliseconds."""
    prefix_rebuild_ms: float = 0.0
    draft_extends_ms: float = 0.0
    target_verify_ms: float = 0.0
    cache_extend_ms: float = 0.0      # v3-only
    tokens_accepted: int = 0

    @property
    def iter_total_ms(self) -> float:
        return (self.prefix_rebuild_ms + self.draft_extends_ms
                + self.target_verify_ms + self.cache_extend_ms)


def v2_iter_cost(prefix_rebuild_ms: float, draft_extends_ms: float,
                 target_verify_ms: float) -> float:
    """Closed form for v2 per-iteration cost (within-iter KV)."""
    return prefix_rebuild_ms + draft_extends_ms + target_verify_ms


def v3_iter_cost(draft_extends_ms: float, target_verify_ms: float,
                 cache_extend_ms: float) -> float:
    """Closed form for v3 per-iteration cost (cross-iter KV).

    The fourth hidden assumption: `cache_extend_ms` is NOT free on
    bandwidth-bound hardware. On T4 we measured 67.2 ms/iter, which
    exceeds the 32.9 ms prefix_rebuild it was designed to eliminate.
    """
    return draft_extends_ms + target_verify_ms + cache_extend_ms


# Hooks for the actual CUDA path. Notebooks bind these to
# HuggingFace/PyTorch models; CLI entry points use the same binding.
DraftFn = Callable[[list[int], Optional[object]], tuple[list[TreeNode], object]]
VerifyFn = Callable[[list[TreeNode], object], tuple[list[int], object]]
