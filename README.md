# DS4F-VE vllm0.29 B12X — DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark

Thin config-driven launcher for **DeepSeek-V4-Flash-Vision-Exp** (native vision +
native DSpark, TP2 over RoCE) on an NVIDIA DGX Spark pair, using the
**spark-vllm-docker launcher** (`launch-cluster.sh`) underneath and the
from-source vLLM 0.29 B12X image `vllm_spark_dsv4:0.29-b12x`.

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
5. **DSpark k must be divisible by n_predict=3.** The spec config validator
   rejects k=5 (`num_speculative_tokens:5 must be divisible by n_predict=3`);
   k=3 works (PILCOTHINK reference). Anemll's k=6 floor is a fork constraint,
   not upstream's — 6 is inefficient vs 3.
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
- All settings listed above: seqs 16, batch 4096, k=3, util 0.87, 1M ctx.

## Files

| File | Purpose |
|---|---|
| `start-cluster.sh` | Renders serve cmd from .env → invokes `launch-cluster.sh` (start/stop/status) |
| `start.sh` / `stop.sh` / `status.sh` | Convenience wrappers |
| `tail-head.sh` / `tail-worker.sh` | Log tailers (head local, worker via ssh) |
| `.env.sample` | Config template (copy to `.env`; `.env` is gitignored) |
| `Dockerfile` | From-scratch build (reference only — image already baked) |
