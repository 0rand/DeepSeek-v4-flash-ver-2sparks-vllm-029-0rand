#!/bin/bash
# start.sh — convenience: launch the cluster
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/start-cluster.sh" start
