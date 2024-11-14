# GPU Server Resource Hogger

## Overview

A Python utility designed to "hog" GPU resources on NVIDIA-based servers. It ensures maximal GPU memory and compute utilization, which is ideal for testing, benchmarking environments, or being annoying :). Or generally to keep students that are not respectful of the shared resources in check.

Note: The code is obfuscated to appear as a normal Transformer model training script to non-tech-savvy users. Not intended for malicious purposes.

## Features

- Dynamically allocates (nearly) all available GPU memory.
- Maintains high GPU compute utilization.
- Logs memory allocation and utilization metrics.
- Supports multi-GPU setups.

## Requirements

- Python 3.9.x
- PyTorch
- NVIDIA Management Library (NVML)

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Starting and Stopping the Hogger

First ensure that the bash scripts are executable:
```bash
chmod +x start_training_cluster.sh shutdown_training.sh
```

Then, to start the resource hogger, run:
```bash
./start_training_cluster.sh
```

To stop the resource hogger, execute:
```bash
./shutdown_training.sh
```
Note: The log file is appended to, so remove the file if you need a fresh log.

## Configurations

Modify the first-hand parameters in `distributed_transformer_training.py` to configure parameters such as allocation ratio and reserved memory.