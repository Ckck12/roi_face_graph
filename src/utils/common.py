# src/utils/common.py

import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """
    시드를 고정하는 함수.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device(rank=0):
    """
    rank를 고려해 CUDA 디바이스를 반환. 
    GPU가 없으면 CPU를 반환.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    else:
        return torch.device("cpu")
