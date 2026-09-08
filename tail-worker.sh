#!/bin/bash
# tail-worker.sh — follow the WORKER node vLLM log (via ssh)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$DIR/.env" ] && { set -a; . "$DIR/.env"; set +a; }
SSH_USER="${SSH_USER:-$(whoami)}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_node}"
IFS=',' read -r _ WIP <<< "${CLUSTER_NODES:-,}"
WIP="$(echo "$WIP" | xargs)"
[ -n "$WIP" ] || { echo "ERROR: cannot determine worker from CLUSTER_NODES" >&2; exit 1; }
exec ssh -o ConnectTimeout=10 "${SSH_USER}@${WIP}" "docker logs -f ${CONTAINER_NAME}" 2>&1
