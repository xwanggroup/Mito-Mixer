# utils.py
import os
# utils.py
import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
