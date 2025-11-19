"""
Module for generating the synthetic arithmetic dataset, as well
as implementing the gymnasium-style environment for RLVR.
"""

"""
TODO: refactor with a generator function and custom operators;
make sure to make it work better and use natural language prompts to
introduce more variability in the dataset. Aim for ~10-20 sample formats
per problem type.

Use 4 digits in each operation. This way we have a large state space of possible samples.
"""

# ==============================
# Imports
# ==============================

import numpy as np
import json

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

# ==============================
# Constants
# ==============================

# TODO: add more problem types as needed (good to start with just these)
PROBLEM_TYPES= {
    "one_digit_addition",
    "two_digit_addition",
    "one_digit_subtraction",
    "two_digit_subtraction",
    "one_digit_multiplication",
    "two_digit_multiplication",
    "one_digit_division",
    "two_digit_division",
}

# ==============================
# Argparsing for Command-Line Usage
# ==============================

# TODO: add argparsing

# ==============================
# Per-Task Generators
# ==============================

# TODO: fix docstrings for the missing ones

def generate_one_digit_addition_samples(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate a list of dictionaries containing one-digit addition problems,
    ready to be added to the exported JSONL file. Uses numpy's random number generator
    for reproducibility, defaults to seed of 42.

    Dictionary structure:
    (
        "input": "What is (x) + (y)",
        "target": (x + y),
        "metadata": (
            "x": x,
            "y": y,
            "type": "one_digit_addition"
        )
    )

    Args:
        num_samples (int): The number of one-digit addition problems to generate.
        seed (int): The seed for the random number generator.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the generated problems.
    """

    rng = np.random.default_rng(seed)

    samples = []

    for _ in range(num_samples):
        x = int(rng.integers(0, 9))
        y = int(rng.integers(0, 9))

        sample = {
            "input": f"What is {x} + {y}?",
            "target": x + y,
            "metadata": {
                "x": x,
                "y": y,
                "type": "one_digit_addition"
            }
        }

        samples.append(sample)
    
    return samples

def generate_two_digit_addition_samples(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate a list of dictionaries containing two-digit addition problems,
    ready to be added to the exported JSONL file. Uses numpy's random number generator
    for reproducibility, defaults to seed of 42.

    Dictionary structure:
    (
        "input": "What is (x) + (y)",
        "target": (x + y),
        "metadata": (
            "x": x,
            "y": y,
            "type": "two_digit_addition"
        )
    )

    Args:
        num_samples (int): The number of two-digit addition problems to generate.
        seed (int): The seed for the random number generator.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the generated problems.
    """

    rng = np.random.default_rng(seed)

    samples = []

    for _ in range(num_samples):
        x = int(rng.integers(10, 99))
        y = int(rng.integers(10, 99))

        sample = {
            "input": f"What is {x} + {y}?",
            "target": x + y,
            "metadata": {
                "x": x,
                "y": y,
                "type": "two_digit_addition"
            }
        }

        samples.append(sample)

    return samples

def generate_one_digit_subtraction_samples(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate one-digit subtraction problems ensuring nonnegative results.
    Example: max(x, y) - min(x, y)
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(num_samples):
        a = int(rng.integers(0, 9))
        b = int(rng.integers(0, 9))
        x, y = max(a, b), min(a, b)

        sample = {
            "input": f"What is {x} - {y}?",
            "target": x - y,
            "metadata": {
                "x": x,
                "y": y,
                "type": "one_digit_subtraction"
            }
        }
        samples.append(sample)

    return samples


def generate_two_digit_subtraction_samples(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate two-digit subtraction problems ensuring nonnegative results.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(num_samples):
        a = int(rng.integers(10, 99))
        b = int(rng.integers(10, 99))
        x, y = max(a, b), min(a, b)

        sample = {
            "input": f"What is {x} - {y}?",
            "target": x - y,
            "metadata": {
                "x": x,
                "y": y,
                "type": "two_digit_subtraction"
            }
        }
        samples.append(sample)

    return samples


def generate_one_digit_multiplication_samples(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate one-digit multiplication problems.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(num_samples):
        x = int(rng.integers(0, 9))
        y = int(rng.integers(0, 9))

        sample = {
            "input": f"What is {x} * {y}?",
            "target": x * y,
            "metadata": {
                "x": x,
                "y": y,
                "type": "one_digit_multiplication"
            }
        }
        samples.append(sample)

    return samples


def generate_two_digit_multiplication_samples(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate two-digit multiplication problems.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(num_samples):
        x = int(rng.integers(10, 99))
        y = int(rng.integers(10, 99))

        sample = {
            "input": f"What is {x} * {y}?",
            "target": x * y,
            "metadata": {
                "x": x,
                "y": y,
                "type": "two_digit_multiplication"
            }
        }
        samples.append(sample)

    return samples


def generate_one_digit_division_samples(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate one-digit division problems ensuring:
    - y != 0
    - x % y == 0  (clean integer division)
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(num_samples):

        # keep sampling until divisible
        while True:
            y = int(rng.integers(1, 9))      # divisor cannot be zero
            x = int(rng.integers(0, 9))
            if y != 0 and (x % y == 0):
                break

        sample = {
            "input": f"What is {x} / {y}?",
            "target": x // y,
            "metadata": {
                "x": x,
                "y": y,
                "type": "one_digit_division"
            }
        }
        samples.append(sample)

    return samples


def generate_two_digit_division_samples(num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate two-digit division problems ensuring:
    - y != 0
    - x % y == 0  (clean integer division)
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(num_samples):

        while True:
            y = int(rng.integers(1, 99))
            x = int(rng.integers(10, 99))
            if y != 0 and (x % y == 0):
                break

        sample = {
            "input": f"What is {x} / {y}?",
            "target": x // y,
            "metadata": {
                "x": x,
                "y": y,
                "type": "two_digit_division"
            }
        }
        samples.append(sample)

    return samples


# ==============================
# Aggregator / Full Dataset Generator
# ==============================

GENERATORS = {
    "one_digit_addition": generate_one_digit_addition_samples,
    "two_digit_addition": generate_two_digit_addition_samples,
    "one_digit_subtraction": generate_one_digit_subtraction_samples,
    "two_digit_subtraction": generate_two_digit_subtraction_samples,
    "one_digit_multiplication": generate_one_digit_multiplication_samples,
    "two_digit_multiplication": generate_two_digit_multiplication_samples,
    "one_digit_division": generate_one_digit_division_samples,
    "two_digit_division": generate_two_digit_division_samples,
}


def generate_arithmetic_dataset(
    output_dir: Path = Path(".data/"),
    samples_per_type: int = 500,
    problem_types: set[str] = {"one_digit_addition"},
    seed: int = 42
) -> None:
    """
    Generates a synthetic arithmetic dataset with the given problem types and saves it to the output directory.
    Uses numpy's random number generator for reproducibility, defaults to seed of 42.

    Args:
        output_dir (Path): The directory to save the dataset to.
        samples_per_type (int): The number of samples to generate for each problem type.
        problem_types (set[str]): The problem types to generate samples for.
        seed (int): The seed for the random number generator.

    Returns:
        None: The dataset is saved to the output directory.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "arithmetic_data.jsonl"

    all_samples = []

    for ptype in (pbar:=tqdm(problem_types)):
        pbar.set_description(f"Generating {ptype} samples...")
        samples = GENERATORS[ptype](samples_per_type, seed)
        all_samples.extend(samples)
    
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=4)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset contains {len(all_samples)} samples.")
    print(f"Sample: {all_samples[0]}")

    return all_samples