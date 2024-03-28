# GPU Server Resource Hogger

## Overview
A Python utility designed to "hog" GPU resources on NVIDIA-based servers. It ensures maximal GPU memory and compute utilization, which is ideal for testing, benchmarking environments, or being annoying :).

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
First, ensure that the bash scripts are executable:
```bash
chmod +x start_hogging.sh stop_hogging.sh
```

Then, to start the resource hogger, run:
```bash
./start_hogging.sh
```

To stop the resource hogger, execute:
```bash
./stop_hogging.sh
```

## Configurations
Modify the first-hand parameters in `gpu_hogging.py` to configure parameters such as allocation ratio and reserved memory.
