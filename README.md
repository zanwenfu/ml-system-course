# Speculative Decoding on Bandwidth-Bound Hardware

**Sequoia predicts 1.68×. I measured 0.56×. A four-term decomposition reconciles the 3× gap to within 1.1% — and reveals a standard optimization that *flips sign* on T4.**

A controlled three-family comparison (linear draft-verify, tree-structured Sequoia, retrieval-based PLD) on a T4 GPU. The paper ports Sequoia's hardware-aware optimizer to T4, predicts the DP-optimal tree, measures the gap, and decomposes it into four independently-measurable hidden assumptions — three in Sequoia's published formula plus a fourth revealed by attempting the natural cross-iteration KV-persistence optimization, which produces a measured slowdown (0.56× → 0.46×) because it replaces a 33 ms prefix rebuild with 67 ms of bandwidth-floored cache extensions.

> Duke CS 590 (ML Systems), Spring 2026. Full technical report: [`CS590_Project_Final_Report.pdf`](./CS590_Project_Final_Report.pdf).

---

## TL;DR

1. **Cost ratio `c = Cd/Ct = 0.596`** — nearly double the parameter ratio of 0.333 — because per-call cost on T4 is dominated by a ~20 ms weight-loading floor from HBM, not by FLOPs.
2. **Sequoia's tree optimizer overpredicts speedup 3×** on T4. Feeding measured per-iteration components back into Sequoia's equation reconciles to **1.1% of the benchmark** — the algorithm is sound; three specific assumptions in the cost model break.
3. **A standard optimization flips sign.** Cross-iteration KV persistence (v3) is a clear win on A100 and a measurable *slowdown* on T4, because each cache-extension forward still pays the bandwidth floor. The fourth hidden assumption, and the paper's sharpest finding.
4. **PLD is the only family that wins on T4** (1.28–1.39×), because `Cr ≈ 0` via CPU n-gram matching is the only structural bypass of the bandwidth floor.

---

## Quick start

```bash
git clone https://github.com/zanwenfu/speculative-decoding-t4
cd speculative-decoding-t4
pip install -r requirements.txt

# Reproduce Table 1 (linear SD break-even curves)
python src/bench_linear.py --draft Llama-3.2-1B --target Llama-3.2-3B --k 3 5 7 10

# Reproduce Sequoia DP optimizer port + v2 tree runtime (§9)
python src/bench_tree.py --tree-config 16x3 --version v2

# Reproduce the fourth-assumption finding (v3 cross-iter KV)
python src/bench_tree.py --tree-config 16x3 --version v3

# Per-call cost probe (§8 — the unifying mechanism)
python src/probe_percall.py --cache-lengths 30 64 128 256
```

Designed to run on a single T4 (16 GB) via Google Colab. Full reproducibility protocol in [§Reproducibility](#reproducibility).

---

## Headline results

### Cross-family speedup on T4 (greedy, Llama-3.2 1B/3B, fp16)

| Family | Configuration | Speedup | Verdict |
|---|---|---:|---|
| **Linear SD** | k=3 | **1.01×** | Marginal; α=0.732 just clears threshold 0.697 |
| Linear SD | k=5 | 0.98× | Crosses into slowdown |
| Linear SD | k=10 | 0.92× | 8% slowdown |
| **Tree SD (v2, within-iter KV)** | n=16, d=3 *(Sequoia DP-optimal)* | 0.56× | Predicted 1.68× |
| Tree SD (v2) | n=8, d=3 | 0.77× | Predicted 1.31× |
| **Tree SD (v3, cross-iter KV)** | n=16, d=3 | **0.46×** | Standard optimization *worsens* v2 |
| **Prompt Lookup Decoding** | n=3 | 1.28× | Only family with consistent speedup |
| Prompt Lookup Decoding | n=7 | 1.35× | — |
| **Prompt Lookup Decoding** | n=10 | **1.39×** | Best result; zero GPU draft cost |

### Sequoia gap decomposition (DP-optimal n=16, d=3)

| Component | Sequoia formula | Measured | Error |
|---|---:|---:|---:|
| Accepted tokens per iter (G) | 4.94 | 2.95 | −40% |
| Verify cost (ms) | 43.3 | 68.9 | +59% |
| Draft cost (ms) | 67.3 | 128.1 | +90% |
| Iter total (ms) | 110.6 | 197.0 | +78% |
| **End-to-end speedup** | **1.68×** | **0.56×** | **−67%** |
| *Calibrated formula (measured terms)* | *0.563×* | *0.557×* | *1.1%* |

Plugging measured per-iteration components back into Sequoia's equation reconciles to within Python-orchestration noise. The gap is not implementation defect or measurement noise — it decomposes cleanly into three structured assumption failures, plus a fourth.

---

## The four hidden assumptions

### 1. Cost scales with FLOPs → Cost scales with bandwidth

Parameter ratio = 0.333; measured `c = 22.42/37.63 = 0.596`. A 1B-fp16 forward pays a ~20 ms weight-loading floor regardless of input length or FLOPs: `2 GB / 300 GB/s ≈ 6.7 ms` for weight streaming, plus ~13 ms fixed per-call overhead (kernel launch, layernorm, RoPE, residuals). Cost ratio `c` must be grounded in wall-clock measurement, not parameter counts.

### 2. Tree verification is ~free → Tree verification is bandwidth-amortized, not compute-amortized

Sequoia assumes `t(n=16) = 1.15 × Ct`: 15% overhead to verify 16 tree nodes versus one token. Measured: `1.83 × Ct`. On A100, compute parallelism makes batched tree-verify nearly free. On T4, memory bandwidth is the bottleneck, so processing more tokens provides minimal amortization. The error is consistent across tree sizes (+59% at n=16, +63% at n=8), confirming a general scaling failure of the `t(n)` approximation.

### 3. Sequential draft calls cost `d × c × Ct` → Each call hits the bandwidth floor

Sequoia predicts 67.3 ms for three sequential draft calls at `c × Ct` each. Measured: 128.1 ms — a full prefix rebuild (32.9 ms) plus four single-token extends (4 × 23.8 ms). The per-call probe (§8) shows every extend pays the same ~20 ms floor regardless of how few tokens it adds.

### 4. Cache maintenance between iterations is free → It costs a full forward pass on T4

**This is the paper's novel finding.** The natural fix for assumption 3 is to persist both draft and target KV caches across iterations (v3), eliminating the per-iteration prefix rebuild. On A100 this is unambiguously a win — extending a cache by ~3 tokens requires negligible compute. I implemented it faithfully and benchmarked:

| Component | v2 (within-iter) | v3 (cross-iter) | Δ |
|---|---:|---:|---:|
| Draft prefix rebuild | 32.9 ms | 0.0 ms | −32.9 |
| Draft extends | 95.2 ms | 106.0 ms | +10.8 |
| Target verify | 68.9 ms | 51.2 ms | −17.7 |
| **Cache extension (D+T)** | 0.0 ms | **67.2 ms** | **+67.2** |
| **Iter total** | 197.0 ms | 224.4 ms | +27.4 |
| **Speedup vs. baseline** | **0.557×** | **0.456×** | **−0.101** |

Each cache-extension forward still pays the bandwidth floor: `Cd + Ct ≈ 55 ms` regardless of how few tokens are appended. The 67.2 ms cache-extension cost *exceeds* the 32.9 ms prefix rebuild it was designed to eliminate. **The optimization's sign flips based purely on whether bandwidth or compute is the binding constraint.**

Both v2 and v3 pass an identical correctness gate: 20/20 token-identical output against greedy autoregressive baseline. v3 is not a buggy implementation — it is a faithful realization of the optimization Sequoia's formula implies, and its measured cost is what the bandwidth floor predicts. A four-term cost model adding `C_ext^D + C_ext^T` reconciles v3 to 8% of measurement.

---

## Per-call cost probe: the unifying mechanism

A single controlled probe — single-token incremental decodes with persistent KV caches at `L ∈ {30, 64, 128, 256}`, 50 trials per length — substantiates every other finding in the paper:

| Cache length L | Cd (ms) | Ct (ms) |
|---:|---:|---:|
| 30 | 20.06 | 43.49* |
| 64 | 20.22 | 35.86 |
| 128 | 20.08 | 34.90 |
| 256 | 19.77 | 35.55 |

*\*L=30 shows warmup effect; excluded from variance estimate.*

**Cd is flat at 20.0 ± 0.2 ms across 8.5× range in cache length (2.3% spread).** Per-call cost does not scale with cache length. Per-call cost does not scale with FLOPs. Per-call cost is a weight-loading floor plus fixed per-call overhead, and nothing else moves it.

This single fact explains every finding:

- **Why `c` compresses toward 1** — bandwidth, not compute
- **Why quantization backfires** (§6) — 4-bit target shrinks `Tw^t` but not `Tw^d`, raising effective `c` despite better parameter ratio
- **Why PLD wins** — `Cr ≈ 0` is the only structural bypass
- **Why Sequoia's three assumptions break** — all three need bandwidth to be negligible
- **Why v3 fails** — cache extensions pay the floor just like prefix rebuilds do

### A methodological note on the probe

Checkpoint 2 of this work reported an apparent in-pipeline per-token draft cost of ~1.5 ms — 14.8× lower than standalone `Cd = 22.42 ms` — and attributed it to "KV-cache persistence amortization." The per-call probe refutes that explanation: cost is flat across cache lengths. The 14.8× reduction was a measurement artifact of how aggregate matmul time gets divided across `k` draft positions in HuggingFace's `assisted_generation`. The real per-call draft cost is ~20 ms, consistent with the bandwidth floor. **A controlled probe is how I caught my own earlier attribution error**, which is why §8 exists as a separate measurement subsystem rather than a derived quantity.

---

## Practitioner rule

A five-point deployment checklist for engineers shipping speculative decoding on cost-constrained hardware:

1. **Measure `c = Cd/Ct` via wall-clock timing.** Not FLOPs. Not parameter ratios. The wall-clock ratio is the one that predicts speedup.
2. **Check the break-even.** If measured chain acceptance `α < (kc + 1)/(k + 1)`, expect a slowdown. On bandwidth-bound hardware where `c → 1`, the required `α` is prohibitively high outside of predictable domains (code, instruction-following).
3. **Prefer PLD when the workload has prompt–output overlap.** Code completion, instruction-following, and summarization benefit most. Open-ended generation gets no lift.
4. **Use linear SD only at k=3**, and only with domain-validated `α > 0.70`. Larger `k` crosses the break-even in most deployments.
5. **Avoid tree methods** unless the deployment can absorb the engineering cost of CUDA graphs and fused kernels that close the algorithmic-vs-implementation gap. The per-call floor is the binding constraint; any tree method that does not address it loses.

This is the single section of the README most likely to survive contact with production. If you take one thing from this work, take this.

---

## Measurement infrastructure

Eight measurement subsystems, each required because existing tools do not expose the per-iteration signal this paper's claims depend on:

| # | Subsystem | What it makes possible |
|---|---|---|
| 1 | α-instrumented linear SD loop | Per-position acceptance measurement (HF's `num_assistant_tokens` reports only aggregates) |
| 2 | Standalone Cd/Ct measurement | Wall-clock anchors for every theoretical prediction |
| 3 | Positional acceptance vector (305 samples, SWOR) | Input to Sequoia's DP tree optimizer |
| 4 | Sequoia DP optimizer port | DP-optimal `(n, d)` selection on measured T4 parameters |
| 5 | Minimal Sequoia-compliant tree runtime (v2) | Tree attention with cover property, branch-level fork/consume-in-place KV semantics |
| 6 | Per-call cost probe across cache lengths | Isolates the bandwidth floor; catches the Checkpoint 2 attribution error |
| 7 | Cross-iter persistent cache runtime (v3) | Evidence for the fourth hidden assumption |
| 8 | PyTorch profiler (OOM-mitigated) | Csys ≈ 0.1 ms validation; cross-method CUDA time attribution |

**Correctness gate:** Both v2 and v3 produce 20/20 token-identical output against the greedy autoregressive baseline. No CUDA graphs, no custom kernels, no batch parallelism — pure algorithmic faithfulness to Sequoia's specification.

---

## Limitations

What this work does *not* claim:

- **No CUDA graph / fused kernel implementation.** Production frameworks (vLLM, TensorRT-LLM) use CUDA graphs to fuse multiple model calls, which would mitigate the per-call fixed overhead `Tf`. The gap from my implementation to a graph-fused one is bounded above by my measured `Tf ≈ 13 ms` per call. My claim is about the *algorithmic cost model's* hidden assumptions, not about what's achievable with systems engineering the cost model doesn't capture.
- **No direct A100 cross-validation.** The v3 "qualitative inversion" prediction for A100 follows from the bandwidth-floor model; I did not run v3 on A100 hardware to confirm it empirically. I stand behind the directional claim and flag the specific unverified prediction.
- **Acceptance vector has wide tails.** 305-sample positional acceptance measurement gives tight bounds at `p1`–`p3` but only 9 / 6 samples at `p4` / `p5`. A 1000-sample re-measurement would tighten the G-prediction error in the gap decomposition, but is unlikely to change the qualitative finding.
- **Batch size 1 throughout.** At larger batches, arithmetic intensity rises and the bandwidth-bound regime weakens. My findings predict progressively smaller cost-model gaps as batch size grows on T4; I did not measure this.
- **Single hardware platform.** Every number is from T4. The closed-form generalization rule (below) extrapolates to A100 but uses Sequoia's reported numbers, not my own measurements.
- **10-prompt evaluation suite.** Emphasizes generation quality across five domains over broad coverage. Larger benchmarks (MT-Bench, HumanEval) would tighten per-prompt variance estimates.

---

## A closed-form heuristic for when the assumptions break

Per-call cost decomposes as `C = Tw + Tf`, where `Tw = (params × 2 bytes) / bandwidth` is weight-loading time and `Tf` is per-call fixed overhead. Then:

```
c = Cd / Ct = (Tw^d + Tf) / (Tw^t + Tf)
```

- When `Tf ≪ Tw`: `c → params_d / params_t` (compute-bound regime; Sequoia's assumptions hold)
- When `Tf ≈ Tw`: `c → 1` (bandwidth-bound regime; speculative decoding breaks)

Order-of-magnitude predictions:

- **T4 + 1B/3B**: `Tw^d ≈ 6.7 ms`, `Tw^t ≈ 20 ms`, `Tf ≈ 13 ms` → `c ≈ 0.60`. Measured: **0.596**. (Matches within noise.)
- **A100 + JF68M/7B**: `Tw^d ≈ 0.09 ms`, `Tw^t ≈ 9 ms`, `Tf ≈ 0.4 ms` → `c ≈ 0.053`. Sequoia reports 0.021. **2× discrepancy**, likely from HuggingFace pipeline overhead absorbed into `Tf` in their measurement.

This is a heuristic, not a precise predictor — at A100 scale the formula is accurate to a factor of ~2, not within-noise. It is useful for *predicting the regime* (bandwidth-bound vs. compute-bound) on unseen hardware, not for predicting a specific speedup. The operational version: if `Tw^t ≫ Tf` on your hardware, tree methods have a chance; otherwise, plan on PLD or linear SD at k=3.

---

## Reproducibility

- **Hardware:** Single T4 (16 GB, ~300 GB/s bandwidth) via Google Colab
- **Stack:** CUDA 12.8, PyTorch 2.10, Transformers 4.45
- **Models:** Llama-3.2-3B (target, fp16), Llama-3.2-1B (draft, fp16)
- **Prompt suite:** 10 prompts across 5 domains (general knowledge, code, summarization, instruction, long-context), 128 generated tokens
- **Protocol:** 30 trials per configuration, 3 warmups, explicit `torch.cuda.synchronize()` barriers, seeds fixed at 42
- **Variation:** ≤5% run-to-run across 30-trial samples; qualitative findings robust to seed and prompt-suite variation
- **Correctness:** Both tree runtimes produce 20/20 token-identical output vs. greedy baseline

---

## Positioning

Prior work on speculative decoding falls into three families that this paper unifies under a single controlled evaluation: linear draft-verify ([Leviathan et al. 2023, ICML]; [Chen et al. 2023]), tree-structured speculation ([SpecInfer, ASPLOS 2024]; [Sequoia, NeurIPS 2024]; single-model variants Medusa, Hydra, EAGLE), and retrieval-based methods ([PLD]; [REST, NAACL 2024]; [SAM-Decoding, 2024]).

**Spec-Bench** [Xia et al., ACL 2024] is the closest empirical survey. It runs on A100 with Vicuna model pairs, reports cross-method comparisons without gap decomposition against any published cost model, and does not include bandwidth-bound hardware. To my knowledge this is the first controlled three-family study on bandwidth-bound hardware that quantitatively decomposes the end-to-end gap against a published cost model, reconciles it to Python-orchestration noise with a calibrated formula, and identifies a fourth cost-model term via attempted optimization.

---

## Citation

```bibtex
@misc{fu2026specdecoding,
  title   = {Speculative Decoding on Bandwidth-Bound Hardware:
             Four Hidden Assumptions Behind a 3× Cost-Model Gap},
  author  = {Fu, Zanwen},
  year    = {2026},
  note    = {Duke CS 590 ML Systems, final project report}
}
```

Full technical report: [`CS590_Project_Final_Report.pdf`](./CS590_Project_Final_Report.pdf) · Author: Zanwen (Ryan) Fu, Duke University · zanwen.fu@duke.edu
