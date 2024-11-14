#!/bin/bash
CLUSTER_DIR=$(dirname "$(readlink -f "$0")")
if [ -f "$CLUSTER_DIR/training.pid" ]; then
    kill $(cat "$CLUSTER_DIR/training.pid")
    rm "$CLUSTER_DIR/training.pid"
    echo "Training cluster shutdown completed"
else
    echo "No active training process found"
fi