"""
Module for loading models (Full-Parameter and LoRA) for experiments.
"""

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import Optional, Union, Dict, Any

# ==============================
# Model Loading Utilities
# ==============================

# ==============================
# Simple MLP for Dry Runs
# ==============================

class SimpleMLPConfig(PretrainedConfig):
    model_type = "simple_mlp"
    
    def __init__(self, vocab_size=50257, hidden_size=1024, num_layers=2, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        super().__init__(**kwargs)

class SimpleBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
    def forward(self, x):
        return x + self.mlp(x)

class SimpleMLP(PreTrainedModel):
    """
    A simple Feedforward MLP for dry runs to verify EoS dynamics.
    Structure: Embedding -> Layer(MLP) * N -> Head
    """
    config_class = SimpleMLPConfig 
    
    def __init__(self, config):
        if isinstance(config, dict):
            config = SimpleMLPConfig(**config)
        super().__init__(config)
        
        self.embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = torch.nn.ModuleList([
            SimpleBlock(config) for _ in range(config.num_layers)
        ])
            
        self.head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.head.weight = self.embed.weight
        
    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed(input_ids)
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
            
        logits = self.head(x)
        
        loss = None
        if labels is not None:
             # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )
    
    def gradient_checkpointing_enable(self, **kwargs):
        pass # No-op for dry run

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

    # Check for dry run argument
    use_mlp_dry_run = kwargs.pop("use_mlp_dry_run", False)
    width = kwargs.pop("dry_run_mlp_width", 1024)
    depth = kwargs.pop("dry_run_mlp_depth", 2)
    
    if use_mlp_dry_run:
        print("!!! DRY RUN MODE ENABLED: USING SIMPLE MLP !!!")
        
        vocab_size = 50304 # Pythia default
        
        config = SimpleMLPConfig(vocab_size=vocab_size, hidden_size=width, num_layers=depth)
        model = SimpleMLP(config)
        
        # Move to device if specified
        if "device_map" in kwargs and kwargs["device_map"] != "auto":
             pass # Handled by caller usually, but here we invoke to()
        
        return model

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
