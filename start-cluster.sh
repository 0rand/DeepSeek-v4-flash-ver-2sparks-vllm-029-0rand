#!/bin/bash
# =============================================================================
# start-cluster.sh — launch DeepSeek-V4-Flash-Vision-Exp (vLLM 0.29 B12X, TP2)
# via the spark-vllm-docker launcher (launch-cluster.sh).
#
# ALL editable config lives in .env (see .env.sample). This script only:
#   1. sources .env,
#   2. renders serve script (vllm serve ...) into $DIR/work/serve-rank.sh,
#   3. calls $SPARK_VLLM_DOCKER/launch-cluster.sh with cluster args.
#
# Usage:
#   ./start-cluster.sh start       (default: -d daemon)
#   ./start-cluster.sh stop
#   ./start-cluster.sh status
#   ./tail-head.sh / ./tail-worker.sh
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

[ -f .env ] || { echo "ERROR: .env missing — cp .env.sample .env && edit" >&2; exit 1; }
set -a; . ./.env; set +a

SPARK_VLLM_DOCKER="${SPARK_VLLM_DOCKER:-$HOME/spark-vllm-docker}"
LAUNCHER="$SPARK_VLLM_DOCKER/launch-cluster.sh"
[ -f "$LAUNCHER" ] || { echo "ERROR: $LAUNCHER not found" >&2; exit 1; }

# defaults (override in .env)
IMAGE="${IMAGE:-vllm_spark_dsv4:0.29-b12x}"
CLUSTER_NODES="${CLUSTER_NODES:-}"
ETH_IF="${ETH_IF:-enp1s0f0np0}"
IB_IF="${IB_IF:-rocep1s0f0,roceP2p1s0f0}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"
PORT="${PORT:-8100}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.87}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-48}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
LIMIT_MM_IMAGES="${LIMIT_MM_IMAGES:-8}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1500}"

[[ -n "$CLUSTER_NODES" ]] || { echo "ERROR: CLUSTER_NODES not set" >&2; exit 1; }

# --- render the serve command (identical on every node; launcher injects ranks) ---
mkdir -p "$DIR/work"
SERVE_SCRIPT="$DIR/work/serve.sh"
cat > "$SERVE_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
vllm serve ${MODEL} \\
    --host 0.0.0.0 \\
    --port ${PORT} \\
    --served-model-name ${SERVED_MODEL_NAME} \\
    --trust-remote-code \\
    --tensor-parallel-size ${TENSOR_PARALLEL} \\
    --kv-cache-dtype ${KV_CACHE_DTYPE} \\
    --block-size ${BLOCK_SIZE} \\
    --max-model-len ${MAX_MODEL_LEN} \\
    --max-num-seqs ${MAX_NUM_SEQS} \\
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \\
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \\
    --enable-prefix-caching \\
    --enable-chunked-prefill \\
    --skip-mm-profiling \\
    --limit-mm-per-prompt '{"image": ${LIMIT_MM_IMAGES}}' \\
    --tokenizer-mode deepseek_v4 \\
    --tool-call-parser deepseek_v4 \\
    --enable-auto-tool-choice \\
    --reasoning-parser deepseek_v4 \\
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \\
    --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"${REASONING_EFFORT}"}' \\
    --moe-backend b12x \\
    --linear-backend b12x \\
    --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \\
    --max-cudagraph-capture-size ${MAX_CUDAGRAPH_CAPTURE_SIZE} \\
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \\
    --speculative-config '{"method":"dspark","model":"${MODEL}","num_speculative_tokens":${NUM_SPECULATIVE_TOKENS},"draft_sample_method":"${DRAFT_SAMPLE_METHOD}","attention_backend":"FLASHINFER_MLA_SPARSE_DSV4","enable_adaptive_verification":false}'
EOF
chmod +x "$SERVE_SCRIPT"
echo "serve cmd rendered -> $SERVE_SCRIPT"
sed -n '3p' "$SERVE_SCRIPT" | cut -c1-100

action="${1:-start}"
case "$action" in
  start)
    echo "=== launching cluster via $LAUNCHER ==="
    "$LAUNCHER" -t "$IMAGE" -n "$CLUSTER_NODES" \
        --eth-if "$ETH_IF" --ib-if "$IB_IF" \
        --launch-script "$SERVE_SCRIPT" -d
    echo "=== waiting for :${PORT}/health (max ${READY_TIMEOUT_S}s) ==="
    for i in $(seq 1 $((READY_TIMEOUT_S / 5))); do
      if curl -s -m 2 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "READY on :${PORT} after ~$((i*5))s"
        exit 0
      fi
      sleep 5
    done
    echo "TIMEOUT — check: ./tail-head.sh and ./tail-worker.sh" >&2; exit 1
    ;;
  stop)
    "$LAUNCHER" stop
    ;;
  status)
    "$LAUNCHER" status 2>&1 || true
    if curl -s -m 3 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      echo "HEALTH: OK on :${PORT}"
    else
      echo "HEALTH: DOWN on :${PORT}"
    fi
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2; exit 1
    ;;
esac
