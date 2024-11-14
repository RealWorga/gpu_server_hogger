#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
kill $(cat "$SCRIPT_DIR/gpu_hogging.pid")
rm "$SCRIPT_DIR/gpu_hogging.pid"
