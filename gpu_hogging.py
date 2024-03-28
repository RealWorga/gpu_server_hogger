import torch
import time
from datetime import datetime
import threading
import pynvml

pynvml.nvmlInit()

LOG_FILE_PATH = "gpu_hogging.log"
ALLOCATION_RATIO = 0.98                 # Ratio of free memory to attempt to allocate
CHECK_INTERVAL = 1                      # How often to check for memory availability in seconds
REDUCED_CHECK_INTERVAL = 60             # Reduced check frequency when near full allocation
TARGET_UTILIZATION = 99                 # Target GPU utilization percentage
RESERVED_MEMORY_MB = 1000               # Amount of memory in MB to reserve for utilization workload

last_utilization = {}                   # Dict to store the last utilization value for each GPU

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
    while True:
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_memory_adjusted = free_memory - (RESERVED_MEMORY_MB * 1024 * 1024)
        tensor_size = int(free_memory_adjusted * ALLOCATION_RATIO // 4)
        if tensor_size > 0:
            tensor = torch.ones(
                (tensor_size,), dtype=torch.float32, device=device)
            allocated_tensors.append(tensor)
            free_memory_after, _ = torch.cuda.mem_get_info()
            allocated_memory = (free_memory - free_memory_after) / 1024 / 1024
            total_allocated_percentage = (
                (total_memory - free_memory_after) / total_memory) * 100
            message = (
                f"Allocated: {allocated_memory:.2f} MB | Total Memory: {total_memory / 1024 / 1024:.2f} MB | Total Hogged: {total_allocated_percentage:.2f}%")
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
    while True:
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(device)

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
