"""
Module storing some utility functions for the project.
"""

# ==============================
# Imports
# ==============================

from pathlib import Path
from typing import List, Dict, Any
import json
import random
import numpy as np
import torch
import os

# ==============================
# Reproducibility
# ==============================

def seed_everything(seed: int = 42) -> None:
    """
    Sets the seed for random, numpy, and torch to ensure reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in torch (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set env var for other libraries that might check it
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Global seed set to {seed}")

# ==============================
# Data Helpers
# ==============================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.

    Args:
        path (Path): The path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the data from the JSONL file.
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data