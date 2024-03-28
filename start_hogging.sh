#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
nohup python "$SCRIPT_DIR/gpu_hogging.py" > /dev/null 2>&1 &
echo $! > "$SCRIPT_DIR/gpu_hogging.pid"
