import os
import json
import pickle
from typing import Dict, Any, List, Optional
try:
    import wandb
except ImportError:
    wandb = None

class LocalLogger:
    def __init__(self, log_dir: str, config: Optional[Dict[str, Any]] = None):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics_jsonl_path = os.path.join(self.log_dir, "metrics.jsonl")
        
        # Clear existing file if it exists (new trial)
        if os.path.exists(self.metrics_jsonl_path):
            os.remove(self.metrics_jsonl_path)
            
        self.use_wandb = False
        if config and config.get("use_wandb"):
            if wandb is None:
                print("Warning: WandB requested but not installed. Disabling WandB.")
                self.use_wandb = False
            else:
                self.use_wandb = True
                wandb.init(
                    project=config.get("wandb_project", "post_training_eos"),
                    entity=config.get("wandb_entity"),
                    name=config.get("wandb_run_name"),
                    group=config.get("wandb_group"),
                    config=config
                )

    def log_config(self, config_dict: Dict[str, Any]):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4, sort_keys=True)

    def log(self, metrics_dict: Dict[str, Any]):
        # Write to JSONL
        with open(self.metrics_jsonl_path, "a") as f:
            f.write(json.dumps(metrics_dict) + "\n")
            
        # Log to WandB
        if self.use_wandb:
            step = metrics_dict.get("step")
            wandb.log(metrics_dict, step=step)

    def finalize(self):
        if self.use_wandb:
            wandb.finish()
