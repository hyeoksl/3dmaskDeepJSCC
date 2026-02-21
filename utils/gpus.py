import pynvml

__all__ = ['getUsableGPUs']

def getUsableGPUs():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    best_gpu_id = None
    max_free_mem = 0
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # GPU 사용률이 0이고, 가용 메모리가 가장 큰 GPU 탐색
        if utilization.gpu == 0 and mem_info.free > max_free_mem:
            max_free_mem = mem_info.free
            best_gpu_id = i
            
    pynvml.nvmlShutdown()
    return best_gpu_id