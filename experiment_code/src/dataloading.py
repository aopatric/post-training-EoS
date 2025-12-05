"""
Module for loading and formatting datasets for Supervised Fine-Tuning (SFT).
Supports Pythia models and instruction-tuning datasets Alpaca and Self-Instruct.
"""

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Dict, List, Optional, Any
import torch

# ==============================
# Constants
# ==============================

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]

# Map dataset names to HF hub IDs
DATASET_CONFIGS = {
    "alpaca": "tatsu-lab/alpaca",
    "self_instruct": "yizhongw/self_instruct", # TODO: might want to swap this source later, check if this is the best one
}

# ==============================
# Formatting Functions
# ==============================

def format_alpaca(example: Dict[str, Any]) -> str:
    """
    Formats an Alpaca example into a prompt.
    """
    if example.get("input", ""):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )

def format_self_instruct(example: Dict[str, Any]) -> str:
    """
    Formats a Self-Instruct example into a prompt.
    Note: The exact fields depend on the specific HF dataset version. 
    Commonly 'prompt' and 'completion'.
    """
    # Adjust fields based on the actual dataset structure if needed
    prompt = example.get("prompt", "")
    completion = example.get("completion", "")
    
    return (
        f"{prompt}\n\n"
        f"{completion}"
    )

FORMATTERS = {
    "alpaca": format_alpaca,
    "self_instruct": format_self_instruct,
}

# ==============================
# Tokenizer Utilities
# ==============================

def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Loads the tokenizer for the specified model.
    Ensures pad_token is set for Pythia (GPT-NeoX).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Pythia/GPT-NeoX usually uses eos_token as pad_token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback if no EOS (unlikely for Pythia)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
    return tokenizer

# ==============================
# Dataset Loading
# ==============================

def get_sft_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    seed: int = 42,
    subsample_size: Optional[int] = None
) -> Dataset:
    """
    Loads and formats an SFT dataset.
    
    Args:
        dataset_name: Name of the dataset ('alpaca', 'self_instruct').
        tokenizer: Tokenizer for formatting.
        max_length: Max sequence length.
        seed: Random seed.
        subsample_size: If provided, selects a random subset of this size.
        
    Returns:
        Dataset: The processed dataset.
    """
    from src.utils import seed_everything
    seed_everything(seed)
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    hf_id = DATASET_CONFIGS[dataset_name]
    formatter = FORMATTERS[dataset_name]
    
        
    print(f"Loading {dataset_name} from {hf_id}...")
    try:
        dataset = load_dataset(hf_id, split="train")
    except Exception as e:
        print(f"Error loading dataset {hf_id}: {e}")
        raise e

    def preprocess_function(examples):
        # examples is a dict of lists: {'instruction': ['...'], 'input': ['...'], ...}
        
        # Reconstruct list of dicts for the formatter
        keys = list(examples.keys())
        batch_size = len(examples[keys[0]])
        formatted_texts = []
        
        for i in range(batch_size):
            ex = {k: examples[k][i] for k in keys}
            formatted_texts.append(formatter(ex))
            
        # Tokenize
        model_inputs = tokenizer(
            formatted_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For Causal LM, labels are usually input_ids. 
        # The trainer will shift them automatically.
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs

    print("Formatting and tokenizing dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Processing {dataset_name}"
    )
    
    return processed_dataset

if __name__ == "__main__":
    # Simple test
    print("Testing dataloading...")
    try:
        tok = get_tokenizer("EleutherAI/pythia-70m") # Use small model for fast test
        ds = get_sft_dataset("alpaca", tokenizer=tok, max_length=128)
        print("Dataset loaded successfully.")
        # Set format to pytorch to get tensors
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        print("Sample input_ids shape:", ds[0]['input_ids'].shape)
        print("Sample decoded:", tok.decode(ds[0]['input_ids'], skip_special_tokens=True))
    except Exception as e:
        print(f"Test failed: {e}")

def get_sft_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 512,
    seed: int = 42,
    subsample_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False
) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for SFT.
    """
    from transformers import default_data_collator
    
    dataset = get_sft_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed,
        subsample_size=subsample_size
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader
