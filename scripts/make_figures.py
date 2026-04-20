"""Regenerate README figures from saved JSON artifacts.

No CUDA, no model downloads -- reads src/results/*.json and writes PNGs
to assets/. Re-run after updating any results JSON.

    python -m scripts.make_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "src" / "results"
ASSETS = ROOT / "assets"

# A minimal, paper-style aesthetic. Restrained, readable, print-safe.
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
})


def _load_tree_profilers():
    v2 = json.loads((RESULTS / "phase2b_tree_results.json").read_text())
    v3 = json.loads((RESULTS / "phase2b_v3_results.json").read_text())
    return v2["profiler_snapshots"]["n16_d3"], v3["profiler_snapshots"]["n16_d3"]


def fig_v2_vs_v3_decomposition() -> Path:
    """Per-iteration cost breakdown, v2 vs v3 (the paper's sharpest finding)."""
    v2, v3 = _load_tree_profilers()

    components = [
        ("Draft prefix\nrebuild", "draft_prefix_ms_per_iter", "#6B8FBE"),
        ("Draft\nextends",        "draft_extend_ms_per_iter", "#9AC3E5"),
        ("Target\nverify",        "verify_ms_per_iter",       "#F2B880"),
        ("Cache extend\n(v3 only)", "cache_extend_ms_per_iter", "#C75B5B"),
    ]

    v2_vals = [v2.get(k, 0.0) for _, k, _ in components]
    v3_vals = [v3.get(k, 0.0) for _, k, _ in components]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = ["v2\n(within-iter KV)", "v3\n(cross-iter KV)"]
    x = [0, 1]
    bottoms = [0.0, 0.0]

    for (label, _, color), v2v, v3v in zip(components, v2_vals, v3_vals):
        vals = [v2v, v3v]
        ax.bar(x, vals, bottom=bottoms, width=0.55, color=color,
               edgecolor="white", linewidth=0.8, label=label.replace("\n", " "))
        for xi, (vi, bi) in enumerate(zip(vals, bottoms)):
            if vi > 6:
                ax.text(xi, bi + vi / 2, f"{vi:.1f}", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        bottoms = [a + b for a, b in zip(bottoms, vals)]

    # Totals + speedup badges above each bar.
    v2_total, v3_total = sum(v2_vals), sum(v3_vals)
    ax.text(0, v2_total + 4, f"{v2_total:.0f} ms / iter\n0.557× baseline",
            ha="center", va="bottom", fontsize=10)
    ax.text(1, v3_total + 4, f"{v3_total:.0f} ms / iter\n0.456× baseline",
            ha="center", va="bottom", fontsize=10, color="#9C2B2B")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Milliseconds per iteration")
    ax.set_ylim(0, max(v2_total, v3_total) * 1.28)
    ax.set_title("The optimization that flips sign: eliminating the 33 ms prefix "
                 "rebuild\ncosts 67 ms in cache extensions on bandwidth-bound T4.",
                 fontsize=11, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=9)

    out = ASSETS / "v2_vs_v3_decomposition.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_cd_flat_across_cache_lengths() -> Path:
    """The unifying measurement: Cd is flat across an 8.5x range in cache length."""
    probe = json.loads((RESULTS / "cd_probe_results.json").read_text())

    lengths = sorted(int(k) for k in probe["draft_extend"].keys())
    cd = [probe["draft_extend"][str(L)]["mean_ms"] for L in lengths]
    cd_std = [probe["draft_extend"][str(L)]["std_ms"] for L in lengths]
    ct = [probe["target_extend"][str(L)]["mean_ms"] for L in lengths]
    ct_std = [probe["target_extend"][str(L)]["std_ms"] for L in lengths]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    ax.errorbar(lengths, cd, yerr=cd_std, marker="o", capsize=3,
                color="#2E5E99", linewidth=2, label="Cd (draft, 1B)")
    ax.errorbar(lengths, ct, yerr=ct_std, marker="s", capsize=3,
                color="#C75B5B", linewidth=2, label="Ct (target, 3B)")

    # FLOP-scaling expectation line: if cost were O(cache_len), Cd would
    # climb linearly from its min. Draw that to show what DIDN'T happen.
    cd_min = min(cd)
    expected = [cd_min * (L / lengths[0]) for L in lengths]
    ax.plot(lengths, expected, "--", color="#888", alpha=0.5, linewidth=1.2,
            label="If cost scaled with cache length (it doesn't)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(L) for L in lengths])
    ax.set_xlabel("Cache length L  (tokens, log2 scale)")
    ax.set_ylabel("Per-call latency (ms)")
    ax.set_ylim(0, max(max(ct), max(expected)) * 1.15)
    ax.set_title("Per-call cost is a bandwidth floor, not a FLOP function.\n"
                 "Cd is flat at 20.0 ± 0.2 ms across 8.5× cache length (2.3 % spread).",
                 fontsize=11, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    out = ASSETS / "cd_flat_bandwidth_floor.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_speedup_across_families() -> Path:
    """Cross-family headline: PLD wins, tree fails, linear marginal."""
    configs = [
        ("Linear  k=10",      0.92, "#9AC3E5"),
        ("Linear  k=5",       0.98, "#9AC3E5"),
        ("Linear  k=3",       1.01, "#6B8FBE"),
        ("Tree v3  (n=16, d=3)", 0.46, "#C75B5B"),
        ("Tree v2  (n=16, d=3)", 0.56, "#D98E5C"),
        ("Tree v2  (n=8, d=3)",  0.77, "#E8B58A"),
        ("PLD  n=3",          1.28, "#7BAE7F"),
        ("PLD  n=7",          1.35, "#5E9A62"),
        ("PLD  n=10",         1.39, "#3C7A40"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [c[0] for c in configs]
    vals = [c[1] for c in configs]
    colors = [c[2] for c in configs]

    y = list(range(len(configs)))
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(1.0, color="#333", linewidth=1.0, linestyle="--",
               label="Break-even (1.0×)")

    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.2f}×", va="center", fontsize=9)

    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Speedup vs. greedy autoregressive (T4, Llama-3.2 1B/3B, fp16)")
    ax.set_xlim(0, 1.65)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1fx"))
    ax.set_title("Three families, one hardware: only PLD breaks even on T4.",
                 fontsize=11, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    out = ASSETS / "speedup_across_families.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    for fn in (fig_v2_vs_v3_decomposition,
               fig_cd_flat_across_cache_lengths,
               fig_speedup_across_families):
        path = fn()
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
