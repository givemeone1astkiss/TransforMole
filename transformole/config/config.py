import torch
import numpy as np
import random
import os

# Path for data.
DATA_PATH='./data/'
# Path for saving models and vocabs.
SAVE_PATH='./save/'
# Path for saving logs.
LOG_PATH='./logs/'
# Path for saving generated SMILES strings.
GEN_PATH='./output/'
# Setting seeds.
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
# Setting device.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"