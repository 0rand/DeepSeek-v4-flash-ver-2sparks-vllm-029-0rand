#!/bin/bash
# stop.sh — stop the cluster (delegates to spark-vllm-docker launcher)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/start-cluster.sh" stop
