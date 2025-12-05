"""
Module for loading models (Full-Parameter and LoRA) for experiments.
"""

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import Optional, Union, Dict, Any

# ==============================
# Model Loading Utilities
# ==============================

def load_base_model(model_name: str, seed: int = 42, **kwargs) -> PreTrainedModel:
    """
    Loads the base causal language model.
    
    Args:
        model_name (str): Name of the model to load.
        seed (int): Random seed for reproducibility.
        **kwargs: Additional arguments for from_pretrained.
        
    Returns:
        PreTrainedModel: The loaded base model.
    """
    from src.utils import seed_everything
    seed_everything(seed)
    
    print(f"Loading base model: {model_name}")
    
    # Default dtype if not in kwargs
    if "torch_dtype" not in kwargs:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        
    # Default device_map if not in kwargs
    if "device_map" not in kwargs:
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **kwargs
    )
    return model

# ==============================
# Full-Parameter Model Wrapper
# ==============================

class FullParamModel:
    """
    Wrapper for loading a model for full-parameter fine-tuning.
    """
    @staticmethod
    def from_pretrained(model_name: str, seed: int = 42, **kwargs) -> PreTrainedModel:
        """
        Loads the model and ensures all parameters are trainable.
        """
        model = load_base_model(model_name, seed, **kwargs)
        
        # Ensure all parameters require grad
        for param in model.parameters():
            param.requires_grad = True
            
        print("Model loaded in Full-Parameter mode.")
        return model

# ==============================
# LoRA Model Wrapper
# ==============================

class LoRAModel:
    """
    Wrapper for loading a model with LoRA adapters.
    """
    @staticmethod
    def from_pretrained(
        model_name: str,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[Union[str, list]] = None,
        seed: int = 42,
        **kwargs
    ) -> PeftModel:
        """
        Loads the base model and applies LoRA configuration.
        """
        model = load_base_model(model_name, seed, **kwargs)
        
        # Default target modules for Pythia/GPT-NeoX if not specified
        if target_modules is None:
            # Common targets for GPT-NeoX architecture (Pythia)
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        
        model = get_peft_model(model, peft_config)
        print(f"Model loaded in LoRA mode with config: r={r}, alpha={lora_alpha}")
        model.print_trainable_parameters()
        
        return model

if __name__ == "__main__":
    # Simple test
    print("Testing model loading...")
    try:
        # Use a very small model for testing
        model_name = "EleutherAI/pythia-70m"
        
        print("\n--- Testing Full Param ---")
        full_model = FullParamModel.from_pretrained(model_name, seed=123)
        print(f"Full param model type: {type(full_model)}")
        
        print("\n--- Testing LoRA ---")
        lora_model = LoRAModel.from_pretrained(model_name, r=4, seed=123)
        print(f"LoRA model type: {type(lora_model)}")
        
    except Exception as e:
        print(f"Test failed: {e}")
