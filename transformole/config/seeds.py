import torch
import numpy as np
import random
import os

# Setting seeds.
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Setting device.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"