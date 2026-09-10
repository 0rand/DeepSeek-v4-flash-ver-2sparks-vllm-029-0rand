# DS4F-VE vllm0.29 B12X — DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark

Thin config-driven launcher for **DeepSeek-V4-Flash-Vision-Exp** (native vision +
native DSpark, TP2 over RoCE) on an NVIDIA DGX Spark pair, using the
**spark-vllm-docker launcher** (`launch-cluster.sh`) underneath and the
from-source vLLM 0.29 B12X image `vllm_spark_dsv4:0.29-b12x`.

Image build instructions by PicloTHINK @ https://github.com/gpdev-Pilcothink/DGX_Spark_vllm_Dockerfile/tree/main/0.28/DSV4F-Vision-exp
## Credits: 
- VLLM team 
- Eugr_nv 
- PilcoThink

Upstream native vision — no monkey-patch mods, no instanttensor, no encoder
file copies.

## Stack

| Component | Pin |
|---|---|
| vLLM | `0.28.1rc1.dev475+g6fbb00b18.d20260907` (6fbb00b1 — native DSV4 vision + DSpark upstream) |
| PyTorch | 2.13.0+cu130 (CUDA 13.0.2 base) |
| FlashInfer | 0.6.18 @ 27d5b029 (DSV4 dual-cache dispatch C4+C128) |
| B12X | 1.3.0 @ 06b4de7c |
| NCCL | SM121 source-built (gencode `compute_121,code=sm_121`) |
| o_proj repair | AST-audited SM12x recipe `(1,128,128)` + packed-scale 3D views, baked in |

Image provenance at runtime: `/workspace/build-metadata.yaml` (inside container).

## Requirements

- Two DGX Spark nodes (GB10, SM121), head + worker, direct RoCE link.
- `vllm_spark_dsv4:0.29-b12x` built via the bundled `Dockerfile`
  (~7.5 h from source, no cache mounts).
- `spark-vllm-docker` operational repo installed on the head
  (`~spark-vllm-docker`, the [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) launcher).
- HF hub tree for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
  (revision **6821d6ad**, 48 shards) present on *both* nodes:
  `$HF_CACHE_DIR/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp`.

## Quick start

```bash
cp .env.sample .env      # edit nodes / paths / serving params
./start-cluster.sh start # launch via spark-vllm-docker launch-cluster.sh
./status.sh              # cluster status + :8100 health
./tail-head.sh           # follow head vLLM log
./stop.sh                # stop both nodes
```

All editable settings live in `.env`. `start-cluster.sh` renders the `vllm serve`
command from it and hands it to `launch-cluster.sh` (`-t IMAGE -n NODES
--eth-if --ib-if --launch-script serve.sh -d`).

## Deliberate serving choices — do NOT regress these

1. **`--kv-cache-dtype fp8` (plain).** `fp8_ds_mla` / `nvfp4_ds_mla` need the
   patched gist stack; on this image they fail with `No valid attention backend`.
   (The log line "Using DeepSeek's fp8_ds_mla KV cache format" is normal — the
   FLASHINFER_MLA_SPARSE_DSV4 backend picks that format internally.)
2. **`--attention-backend FLASHINFER_MLA_SPARSE_DSV4`** (NOT `B12X`).
3. **`VLLM_USE_AOT_COMPILE=0` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1`** — the image
   preflight hard-requires both.
4. **`gpu_memory_utilization` ≤ 0.877.** GB10 reports only ~106.77/121.69 GiB
   free at boot; the README's 0.9 crashes with
   `Free memory ... less than desired GPU memory utilization`. 0.87 is
   tight-but-valid; 0.85 is the proven value.
5. **DSpark k = 6 (not 3).** The spec config validator requires k divisible by
   n_predict=3 (rejects k=5: `num_speculative_tokens:5 must be divisible by
   n_predict=3`), but k=3 is NOT quality-neutral: Primo 2026-09-09 — k=3
   mangled the thinking process (acceptance looked fine, sampled reasoning
   degraded); switching to k=6 improved quality dramatically. Anemll's k=6
   floor is a fork constraint that happens to match the quality optimum.
6. **`--skip-mm-profiling` + `--limit-mm-per-prompt {"image": 8}`** — RAM guard.
   Images in USER messages only (DeepSeek contract; system/assistant images are
   rejected 400). Max 384 image tokens/image.
7. **`max_num_batched_tokens` 4096** (Primo's batch-vs-quality law: 8192→4096
   improves quality by avoiding batch smear across speculation paths; small
   throughput surrender accepted). vLLM warns `max_num_scheduled_tokens is set
   to 4096 based on speculative decoding` — expected, not an error.

## Verified baseline (2026-09-08, Canglong cluster)

- Boot ~10 min (weights 48/48 in 29 s, CUDA graph capture after).
- Text smoke + native vision (dragon portrait) PASS; DSpark acceptance
  per-position ~0.70/0.49/0.28, avg 44–76%; ~56 tok/s on a 600-token gen.
- All settings listed above: seqs 8, batch 4096, k=6, util 0.87, 1M ctx.

## Files

| File | Purpose |
|---|---|
| `start-cluster.sh` | Renders serve cmd from .env → invokes `launch-cluster.sh` (start/stop/status) |
| `start.sh` / `stop.sh` / `status.sh` | Convenience wrappers |
| `tail-head.sh` / `tail-worker.sh` | Log tailers (head local, worker via ssh) |
| `.env.sample` | Config template (copy to `.env`; `.env` is gitignored) |
| `Dockerfile` | From-scratch build (reference only — image already baked) |

## Performance testing

#### Optimal settings -> max batch = 2048, as raising it did not improve DSpark acceptance on synthetic llama-benchy data (set to 3 tokens) ~ 50% but ate half kv cache allocation

### max_batch_tokens=2048 
GMU=0.87
KV Cache = 3.9M tokens
Max SEQ = 8
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test                                                       ┃       c        ┃                      pp t/s ┃                     tg t/s ┃                     TTFT (ms) ┃                   Total (ms) ┃                       Tokens ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ pp1024 tg256 @ d0                                          │       c1       │                       2,744 │                       35.8 │                           664 │                        7,564 │                     1024+256 │
│ pp1024 tg256 @ d0                                          │       c2       │                       1,709 │                       69.4 │                         1,147 │                        7,764 │                     1024+256 │
│ pp1024 tg256 @ d0                                          │       c4       │                       1,856 │                       76.1 │                         1,789 │                       13,544 │                     1024+256 │
│ pp1024 tg256 @ d0                                          │       c8       │                       2,051 │                      107.2 │                         2,808 │                       19,050 │                     1024+256 │
│ pp1024 tg256 @ d4096                                       │       c1       │                       2,100 │                       42.1 │                         2,740 │                        8,569 │                     1024+256 │
│ pp1024 tg256 @ d4096                                       │       c2       │                       1,922 │                       43.5 │                         4,215 │                       13,818 │                     1024+256 │
│ pp1024 tg256 @ d4096                                       │       c4       │                       1,971 │                       52.3 │                         6,956 │                       21,211 │                     1024+256 │
│ pp1024 tg256 @ d4096                                       │       c8       │                       1,966 │                       60.1 │                        12,272 │                       33,988 │                     1024+256 │
│ pp1024 tg256 @ d16384                                      │       c1       │                       1,965 │                       37.1 │                         9,165 │                       15,812 │                     1024+256 │
│ pp1024 tg256 @ d16384                                      │       c2       │                       1,235 │                       18.9 │                        18,831 │                       30,867 │                     1024+256 │
│ pp1024 tg256 @ d16384                                      │       c4       │                       1,685 │                       23.0 │                        24,045 │                       46,231 │                     1024+256 │
│ pp1024 tg256 @ d16384                                      │       c8       │                       1,790 │                       24.7 │                        41,537 │                       76,700 │                     1024+256 │
└────────────────────────────────────────────────────────────┴────────────────┴─────────────────────────────┴────────────────────────────┴───────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test                                                         ┃       c        ┃                     pp t/s ┃                     tg t/s ┃                    TTFT (ms) ┃                   Total (ms) ┃                       Tokens ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ pp1024 tg256 @ d0                                            │       c1       │                      2,479 │                       43.3 │                          713 │                        6,365 │                     1024+256 │
│ pp1024 tg256 @ d4096                                         │       c1       │                      2,100 │                       39.9 │                        2,745 │                        8,904 │                     1024+256 │
│ pp1024 tg256 @ d16384                                        │       c1       │                      2,010 │                       39.5 │                        8,969 │                       15,189 │                     1024+256 │
│ pp1024 tg256 @ d65536                                        │       c1       │                      1,870 │                       39.4 │                       35,905 │                       42,142 │                     1024+256 │
│ pp1024 tg256 @ d128000                                       │       c1       │                      1,760 │                       45.7 │                       73,626 │                       78,961 │                     1024+256 │
└──────────────────────────────────────────────────────────────┴────────────────┴────────────────────────────┴────────────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘

```

### max_batch_tokens=8192
Max SEQ=16
GMU=0.87
KV Cache = 1.5M

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test                                                     ┃       c        ┃                      pp t/s ┃                      tg t/s ┃                     TTFT (ms) ┃                    Total (ms) ┃                       Tokens ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ pp1024 tg256 @ d0                                        │       c1       │                       2,716 │                        43.1 │                           670 │                         6,350 │                     1024+256 │
│ pp1024 tg256 @ d0                                        │       c2       │                       1,567 │                        56.5 │                         1,050 │                         9,366 │                     1024+256 │
│ pp1024 tg256 @ d0                                        │       c4       │                       2,124 │                        71.1 │                         1,750 │                        13,537 │                     1024+256 │
│ pp1024 tg256 @ d0                                        │      c16       │                         895 │                        95.3 │                         6,428 │                        40,093 │                     1024+256 │
│ pp1024 tg256 @ d4096                                     │       c1       │                       2,400 │                        34.6 │                         2,430 │                         9,567 │                     1024+256 │
│ pp1024 tg256 @ d4096                                     │       c2       │                       1,989 │                        49.8 │                         4,683 │                        13,843 │                     1024+256 │
│ pp1024 tg256 @ d4096                                     │       c4       │                       2,187 │                        56.2 │                         6,917 │                        21,024 │                     1024+256 │
│ pp1024 tg256 @ d4096                                     │      c16       │                       2,162 │                        68.8 │                        22,031 │                        56,890 │                     1024+256 │
└──────────────────────────────────────────────────────────┴────────────────┴─────────────────────────────┴─────────────────────────────┴───────────────────────────────┴───────────────────────────────┴──────────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test                                                         ┃       c        ┃                     pp t/s ┃                     tg t/s ┃                    TTFT (ms) ┃                   Total (ms) ┃                       Tokens ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ pp1024 tg256 @ d0                                            │       c1       │                      2,267 │                       37.4 │                          759 │                        7,330 │                     1024+256 │
│ pp1024 tg256 @ d4096                                         │       c1       │                      2,513 │                       37.5 │                        2,341 │                        8,896 │                     1024+256 │
│ pp1024 tg256 @ d16384                                        │       c1       │                      2,146 │                       39.3 │                        8,421 │                       14,669 │                     1024+256 │
│ pp1024 tg256 @ d65536                                        │       c1       │                      1,990 │                       39.0 │                       33,768 │                       40,063 │                     1024+256 │
│ pp1024 tg256 @ d128000                                       │       c1       │                      1,826 │                       36.7 │                       70,990 │                       77,687 │                     1024+256 │
└──────────────────────────────────────────────────────────────┴────────────────┴────────────────────────────┴────────────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘

```
## Tool-call testing
```
1 Seq avg speed on mixed text/json/tools = ~45-50 t/s
16 Seqs = ~145-150 t/s
Dspark acceptance at high concurrency = ~ 70%
At 1 seq = ~ 60%

Configuration: DSpark tokens=3, batch tokens = 8200, GMU = 0.87 (1.9M kv cache)

ker_TP0 pid=257) INFO 09-09 20:24:26 [gpu_worker.py:681] Available KV cache memory: 19.06 GiB
(Worker_TP0 pid=257) INFO 09-09 20:24:26 [gpu_worker.py:696] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.8700 is equivalent to --gpu-memory-utilization=0.8577 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.8823. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=184) INFO 09-09 20:24:26 [kv_cache_utils.py:2312] GPU KV cache size: 1,844,583 tokens, Maximum concurrency for 1,048,576 tokens per request: 1.76x
(Worker_TP0 pid=257) INFO 09-09 20:24:26 [gpu_worker.py:781] Cleared 0.15 GiB of cached CUDA allocator memory before KV cache allocation.


8 seqs ~ 110 t/s
1 seq ~ 45 t/s
Tool-eval-bench scores:
temperature = 0.6, top_p = 0.85
╭─────────────────────────────────────────────────────────────────────────────────────────────────────── 🏆 Benchmark Complete ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                                                                      │
│    Model:  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp                                                                                                                                                                                  │
│    Score:  91 / 100                                                                                                                                                                                                                  │
│    Rating: ★★★★★ Excellent                                                                                                                                                                                                           │
│    Benchmark: tool-eval-bench v2.6.1.dev25+g4365b9031                                                                                                                                                                                │
│    Engine:       vLLM 0.28.1rc1.dev475+g6fbb00b18.d20260907                                                                                                                                                                          │
│    Max context:  1,048,576 tokens                                                                                                                                                                                                    │
│                                                                                                                                                                                                                                      │
│    ✅ 74 passed   ⚠️  12 partial   ❌ 2 failed                                                                                                                                                                                       │
│    Points: 160/176                                                                                                                                                                                                                   │
│                                                                                                                                                                                                                                      │
│    Quality:        91/100                                                                                                                                                                                                            │
│    Responsiveness: 14/100  (median turn: 9.9s)                                                                                                                                                                                       │
│    Deployability:  68/100  (α=0.7)                                                                                                                                                                                                   │
│    Weakest: M Autonomous Planning (67%)                                                                                                                                                                                              │
│                                                                                                                                                                                                                                      │
│    Completed in 548.1s                                                                                                                                                                                                               │
│                                                                                                                                                                                                                                      │
│    📊 Token Usage:                                                                                                                                                                                                                   │
│    Total: 593,063 tokens  │  Efficiency: 0.3 pts/1K tokens                                                                                                                                                                           │
│                                                                                                                                                                                                                                      │
│    ── How this score is calculated ──                                                                                                                                                                                                │
│    • Each scenario: pass=2pt, partial=1pt, fail=0pt                                                                                                                                                                                  │
│    • Category %: earned / max per category                                                                                                                                                                                           │
│    • Final score: (total points / max points) × 100                                                                                                                                                                                  │
│    • Deployability: 0.7×quality + 0.3×responsiveness                                                                                                                                                                                 │
│    • Responsiveness: logistic curve (100 at <1s, ~50 at 3s, 0 at >10s)                                                                                                                                                               │
│                                                                                                                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```


