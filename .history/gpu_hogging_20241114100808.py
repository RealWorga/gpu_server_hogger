import torch
import time
from datetime import datetime
import threading
import pynvml

pynvml.nvmlInit()

# Non-adjustable constant parameters
BYTES_PER_KB = 1024

# Adjustable constant parameters
LOG_FILE_PATH = "gpu_hogging.log"
ALLOCATION_RATIO = 0.94             # Ratio of free memory to attempt to allocate
CHECK_INTERVAL = 1                  # How often to check for memory availability in seconds
REDUCED_CHECK_INTERVAL = 60         # Reduced check frequency when near full allocation
GPU_WORKLOAD_SIZE = 100             # Initial workload size for GPU Util, (most likely to need revision based on your GPU(s))
RESERVED_MEMORY_UTIL = 1000         # Initial memory to reserve for utilization workload

last_utilization = {}

def log_message(gpu_id, message_type, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"{timestamp} | GPU {gpu_id} | {message_type} | {message}\n"
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(formatted_message)

def hog_gpu_memory(gpu_id):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    allocated_tensors = []
    near_full_allocation = False
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    while True:
        free_memory, _ = torch.cuda.mem_get_info()
        free_memory_adjusted = free_memory - \
            (RESERVED_MEMORY_UTIL * BYTES_PER_KB ** 2)
        tensor_size = int(free_memory_adjusted * ALLOCATION_RATIO // 4)
        if tensor_size > 0:
            tensor = torch.ones(
                (tensor_size,), dtype=torch.float32, device=device)
            allocated_tensors.append(tensor)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            allocated_memory = (mem_info.used) / BYTES_PER_KB ** 2
            total_allocated_percentage = (mem_info.used / total_memory) * 100
            message = (
                f"Allocated: {allocated_memory:.2f} MiB | Total Memory: {total_memory / BYTES_PER_KB ** 2:.2f} MiB | Total Hogged: {total_allocated_percentage:.2f}%")
            log_message(gpu_id, "Memory", message)
            near_full_allocation = total_allocated_percentage > 95
        else:
            if not near_full_allocation:
                log_message(
                    gpu_id,
                    "Memory",
                    "Allocation reached near full capacity, reducing check frequency.")
                near_full_allocation = True
        time.sleep(
            REDUCED_CHECK_INTERVAL if near_full_allocation else CHECK_INTERVAL)

def get_gpu_utilization(gpu_id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return utilization.gpu

def continuous_workload(device):
    torch.cuda.set_device(device)
    workload_size = GPU_WORKLOAD_SIZE
    while True:
        try:
            a = torch.randn(workload_size, workload_size, device=device)
            b = torch.randn(workload_size, workload_size, device=device)
            c = torch.matmul(a, b)
            free_memory, _ = torch.cuda.mem_get_info(device)
            if free_memory > workload_size * workload_size * 4 * 3:
                workload_size = int(workload_size * 1.1)
            else:
                workload_size = max(100, int(workload_size * 0.9))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                workload_size = max(100, int(workload_size * 0.9))
                torch.cuda.empty_cache()
            else:
                raise e

def hog_gpu_utilization(gpu_id):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    log_message(gpu_id, "Utilization", "Starting utilization hogging.")
    threading.Thread(target=continuous_workload,
                     args=(device,), daemon=True).start()

    global last_utilization
    last_utilization[gpu_id] = -1

    while True:
        current_utilization = get_gpu_utilization(gpu_id)
        if current_utilization != last_utilization[gpu_id]:
            log_message(gpu_id, "Utilization",
                        f"Current utilization: {current_utilization}%.")
            last_utilization[gpu_id] = current_utilization
        time.sleep(CHECK_INTERVAL)

num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    threading.Thread(target=hog_gpu_memory, args=(i,), daemon=True).start()
    threading.Thread(target=hog_gpu_utilization,
                     args=(i,), daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    log_message(-1, "Service", "Stopping GPU hogging...")
    pynvml.nvmlShutdown()
