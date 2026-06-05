import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.memory_allocated(0) / 1024**3
    print(f'Allocated: {mem:.2f} GB')
    torch.cuda.empty_cache()
    print(f'After empty_cache: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
