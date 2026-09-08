#!/bin/bash
# tail-head.sh — follow the HEAD node vLLM log
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$DIR/.env" ] && { set -a; . "$DIR/.env"; set +a; }
CONTAINER_NAME="${CONTAINER_NAME:-vllm_node}"
exec docker logs -f "$CONTAINER_NAME" 2>&1
