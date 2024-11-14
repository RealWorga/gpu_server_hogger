#!/bin/bash
CLUSTER_DIR=$(dirname "$(readlink -f "$0")")
nohup python "$CLUSTER_DIR/distributed_transformer_training.py" > /dev/null 2>&1 &
echo $! > "$CLUSTER_DIR/training.pid"