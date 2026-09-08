#!/bin/bash
# status.sh — cluster status + health
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/start-cluster.sh" status
