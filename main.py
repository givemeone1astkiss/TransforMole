from torch.cuda import is_available, device_count
from torch import cuda, device
import torch

print(is_available())
print(device_count())
print(device(f"cuda:{cuda.current_device()}"))

A = torch.tensor([1, 2, 3]).to(device(f"cuda:{cuda.current_device()}"))