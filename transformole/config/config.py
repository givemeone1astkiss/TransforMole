import torch
import numpy as np
import random

# Path for data.
DATA_PATH='./data/'
# Path for saving models and vocabs.
SAVE_PATH='./save/'
# Setting seeds.
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
