import yaml
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Union, Any
import os

LIST_FIELDS = ["lora_target_modules", "sharpness_layer_indices"]

@dataclass
class ExperimentConfig:
    # identity
    experiment_name: Optional[str] = None
    output_dir: str = "results"
    seed: int = 42
    
    # wandb
    use_wandb: bool = False
    wandb_project: str = "post_training_eos"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    
    # model
    model_name: str = "EleutherAI/pythia-2.8b"
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # data
    dataset_name: str = "alpaca"
    max_length: int = 2048
    subsample_size: Optional[int] = None
    
    # training/opt
    batch_size: int = 8
    eval_batch_size: int = 2 # Small batch for expensive metrics
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    momentum: float = 0.0
    max_steps: int = 1000
    eval_interval: int = 100
    save_interval: int = 500
    
    # misc
    device: str = "cuda"
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    max_grad_norm: float = 1.0
    compute_spectral_sharpness: bool = True
    compute_batch_sharpness: bool = False
    compute_global_sharpness: bool = True
    compute_block_sharpness: bool = True
    sharpness_layer_indices: Optional[List[int]] = None

    def __post_init__(self):
        # force cast types
        if self.learning_rate is not None:
            self.learning_rate = float(self.learning_rate)
        if self.batch_size is not None:
            self.batch_size = int(self.batch_size)
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        if self.eval_interval is not None:
            self.eval_interval = int(self.eval_interval)
        if self.save_interval is not None:
            self.save_interval = int(self.save_interval)
        if self.weight_decay is not None:
            self.weight_decay = float(self.weight_decay)
        if self.momentum is not None:
            self.momentum = float(self.momentum)
        if self.lora_r is not None:
            self.lora_r = int(self.lora_r)
        if self.lora_alpha is not None:
            self.lora_alpha = int(self.lora_alpha)
        if self.lora_dropout is not None:
            self.lora_dropout = float(self.lora_dropout)

    
    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "ExperimentConfig":
        """
        Generate an ExperimentConfig from a dict of keys.
        """
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "ExperimentConfig":
        """
        """
        with open(filepath, "r") as f:
            raw_cfg = yaml.safe_load(f)
        return cls.from_dict(raw_cfg)

def expand_config(filepath: str) -> List[ExperimentConfig]:
    """
    Expands a YAML config file into a list of ExperimentConfig objects.
    Supports "zipped" expansion where lists of values are paired 1-to-1.
    """
    import copy
    
    with open(filepath, "r") as f:
        raw_cfg = yaml.safe_load(f)
        
    # separate keys with list values from keys with single values
    list_keys = []
    list_values = []
    single_keys = {}
    
    for k, v in raw_cfg.items():
        if isinstance(v, list) and not (k in LIST_FIELDS): # screen for list field params
            list_keys.append(k)
            list_values.append(v)
        else:
            single_keys[k] = v
            
    # just do normal init if no sweeps
    if not list_keys:
        return [ExperimentConfig.from_dict(single_keys)]

    # prevent misshaped configs
    list_lengths = [len(v) for v in list_values]
    if len(set(list_lengths)) > 1:
        raise ValueError(f"All list arguments in config must have the same length. Found lengths: {dict(zip(list_keys, list_lengths))}")
    
    num_configs = list_lengths[0]
    configs = []
    
    # zipped expansion
    for i in range(num_configs):
        current_cfg_dict = copy.deepcopy(single_keys)
        
        # Build run name suffix from varied params
        run_name_parts = []
        
        for k, v_list in zip(list_keys, list_values):
            val = v_list[i]
            current_cfg_dict[k] = val
            
            # Format value for run name (shorten floats)
            if isinstance(val, float):
                val_str = f"{val:.2e}" if val < 1e-3 or val > 1e3 else f"{val}"
            else:
                val_str = str(val)
            run_name_parts.append(f"{k}={val_str}")
            
        # Set wandb_run_name
        varied_suffix = ",".join(run_name_parts)
        if "wandb_run_name" in current_cfg_dict and current_cfg_dict["wandb_run_name"]:
             current_cfg_dict["wandb_run_name"] = f"{current_cfg_dict['wandb_run_name']}-{varied_suffix}"
        else:
             current_cfg_dict["wandb_run_name"] = varied_suffix
            
        configs.append(ExperimentConfig.from_dict(current_cfg_dict))
        
    return configs