"""Per-call cost probe: the unifying measurement in the paper.

Given a saved probe JSON produced by `src/notebooks/phase2b_with_cd_probe.ipynb`
or by `src/bench/probe_percall.py`, this module summarises the bandwidth-floor
finding that Cd is flat across cache length.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterable


def per_call_probe_summary(results_path: str | Path) -> dict:
    """Summarise a per-call probe JSON.

    Expects the schema produced by phase2b_with_cd_probe.ipynb: top-level
    `draft_extend` and `target_extend` dicts keyed by cache length.
    Returns per-length Cd/Ct and the aggregate spread that underwrites
    the README's 'bandwidth floor' claim.
    """
    data = json.loads(Path(results_path).read_text())
    draft = data["draft_extend"]
    target = data["target_extend"]

    cd_means, ct_means = [], []
    rows = {}
    for length_str, row in draft.items():
        length = int(length_str)
        cd = row["mean_ms"]
        ct = target[length_str]["mean_ms"]
        cd_means.append(cd)
        ct_means.append(ct)
        rows[length] = {"Cd_ms": cd, "Ct_ms": ct}

    cd_spread = (max(cd_means) - min(cd_means)) / statistics.mean(cd_means)
    ct_spread = (max(ct_means) - min(ct_means)) / statistics.mean(ct_means)

    return {
        "per_length": rows,
        "Cd_mean_ms": statistics.mean(cd_means),
        "Cd_std_ms": statistics.pstdev(cd_means),
        "Cd_relative_spread": cd_spread,
        "Ct_mean_ms": statistics.mean(ct_means),
        "Ct_relative_spread": ct_spread,
        "finding": (
            "Cd flat at %.2f +/- %.2f ms across %.1fx range in cache length"
            " (%.1f%% spread)." % (
                statistics.mean(cd_means),
                statistics.pstdev(cd_means),
                max(rows) / min(rows),
                cd_spread * 100,
            )
        ),
    }


def fit_fixed_overhead(cd_measurements: Iterable[float], params_draft: float,
                       bandwidth_gbps: float, bytes_per_param: int = 2) -> float:
    """Invert C = Tw + Tf to recover Tf from measured Cd.

    Useful when bandwidth is known (T4 HBM ~ 300 GB/s) but Tf isn't
    directly measurable. Given a fleet of Cd readings across cache
    lengths, the floor is their min; Tf = min(Cd) - Tw.
    """
    cd_min = min(cd_measurements)
    tw_d = (params_draft * bytes_per_param) / (bandwidth_gbps * 1e9) * 1e3
    return cd_min - tw_d
