"""CLI entry points. Thin wrappers over `src.specdec`; each is intended
to be run via `python -m src.bench.<name>`.

  * bench_linear  - reproduce linear-SD break-even analysis
  * bench_tree    - reproduce Sequoia DP + v2/v3 gap decomposition
  * probe_percall - summarise the per-call cost probe (bandwidth floor)
  * advise        - deployable advisor: pick a family for a given hw/model pair
"""
