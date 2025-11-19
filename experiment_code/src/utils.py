"""
Module storing some utility functions for the project.
"""

# ==============================
# Imports
# ==============================

from pathlib import Path
from typing import List, Dict, Any
import json

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