"""
Script for running Supervised Fine-Tuning (SFT) experiments with Edge of Stability (EoS) metric tracking.
"""

import argparse
import os
import sys
import logging
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig, expand_config
from src.models import FullParamModel, LoRAModel
from src.dataloading import get_sft_dataloader
from src.utils import seed_everything
from src.metrics import compute_sharpness
from src.logger import LocalLogger

def train(config: ExperimentConfig):
    # Setup Output Directory
    if not config.experiment_name:
        import uuid
        config.experiment_name = f"sft_{str(uuid.uuid4())[:8]}"
        
    output_dir = Path(config.output_dir) / config.experiment_name
    
    # Clean existing results directory if it exists
    if output_dir.exists():
        print(f"Removing existing results directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Logger
    logger = LocalLogger(str(output_dir), config.to_dict())
    logger.log_config(config.to_dict())
    
    print(f"Starting experiment: {config.experiment_name}")
    print(f"Config: {config}")
    
    # Seed Everything
    seed_everything(config.seed)
    
    # Load Model
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with eager attention for EoS metrics (HVP support)
    # Force FP32 for stability in sharpness computation
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.float32,
        "use_mlp_dry_run": config.use_mlp_dry_run,
        "dry_run_mlp_width": config.dry_run_mlp_width,
        "dry_run_mlp_depth": config.dry_run_mlp_depth
    }
    
    if config.use_lora:
        model = LoRAModel.from_pretrained(
            config.model_name,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            seed=config.seed,
            **model_kwargs
        )
    else:
        model = FullParamModel.from_pretrained(
            config.model_name, 
            seed=config.seed,
            **model_kwargs
        )
    
    model.to(device)
    
    if config.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Dataloader
    # 1. Load Dataset ONCE
    from src.dataloading import get_sft_dataset
    
    full_dataset = get_sft_dataset(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        max_length=config.max_length,
        seed=config.seed,
        subsample_size=config.subsample_size
    )
    
    # 2. Create Train Dataloader
    dataloader = get_sft_dataloader(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        seed=config.seed,
        subsample_size=config.subsample_size,
        dataset=full_dataset # Reuse
    )
    
    # Optimizer
    if config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Create a fixed batch for EoS metrics to ensure consistency (reduce noise from data)
    # print("Creating fixed validation batch for EoS metrics...")
    
    # Reuse the SAME dataset object, just select a small subset for the eval loader
    # This avoids reloading/retokenizing
    # eval_subset = full_dataset.select(range(min(100, len(full_dataset))))
    
    # eval_dataloader = get_sft_dataloader(
    #     dataset_name=config.dataset_name,
    #     tokenizer=tokenizer,
    #     batch_size=config.eval_batch_size, # Use separate small batch size for eval
    #     max_length=config.max_length,
    #     seed=config.seed + 1, 
    #     dataset=eval_subset # Reuse subset
    # )
    # eval_batch = next(iter(eval_dataloader))
    # eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
    
    # Training Loop
    model.train()
    progress_bar = tqdm(range(config.max_steps))
    data_iter = iter(dataloader)
    
    step = 0
    micro_step = 0
    optimizer.zero_grad() # Initialize gradients
    
    while step < config.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # --- EoS Metric Tracking ---
        # Only run metrics at the start of an optimization step
        if step % config.eval_interval == 0 and micro_step % config.gradient_accumulation_steps == 0:
            print(f"Step {step}: Computing sharpness...")
            
            # Define loss function wrapper for metrics
            def loss_wrapper(m, b):
                return m(**b).loss
            
            # 1. Identify blocks and create mapping for friendly names
            block_mapping = {} # friendly_name -> actual_name
            
            # Find the layers container (usually 'gpt_neox.layers' or similar)
            layers_prefix = None
            for name, module in model.named_modules():
                if name.endswith("layers") and isinstance(module, torch.nn.ModuleList):
                    layers_prefix = name
                    break
            
            if layers_prefix:
                # Iterate through layers
                # We access the module list to get the length
                num_layers = len(dict(model.named_modules())[layers_prefix])
                # Actually easier to just iterate named_modules again or assume structure
                # Let's just find them by name pattern to be safe
                
                for name, module in model.named_modules():
                    if name.startswith(layers_prefix) and name.count('.') == layers_prefix.count('.') + 1:
                        # This is a layer, e.g. gpt_neox.layers.0
                        try:
                            idx = int(name.split('.')[-1])
                            friendly_layer = f"layer_{idx}"
                            block_mapping[friendly_layer] = name
                            
                            # Look for sub-modules
                            for child_name, _ in module.named_children():
                                if "attention" in child_name:
                                    block_mapping[f"{friendly_layer}_attn"] = f"{name}.{child_name}"
                                elif "mlp" in child_name:
                                    block_mapping[f"{friendly_layer}_mlp"] = f"{name}.{child_name}"
                        except ValueError:
                            continue
            else:
                # Fallback: look for any "layers.X"
                for name, _ in model.named_modules():
                    if "layers." in name and name.split(".")[-1].isdigit():
                        idx = name.split(".")[-1]
                        block_mapping[f"layer_{idx}"] = name
            
            actual_block_names = list(block_mapping.values())
            
            # Filter blocks if sharpness_layer_indices is set
            if config.sharpness_layer_indices is not None:
                # Identify all unique layer indices
                # We assume friendly names are like "layer_N" or "layer_N_suffix"
                # We want to keep blocks that belong to the requested layers
                
                # First, find max layer index to handle negative indices
                max_layer_idx = -1
                for name in block_mapping.keys():
                    if name.startswith("layer_"):
                        try:
                            # extract N from layer_N...
                            parts = name.split('_')
                            idx = int(parts[1])
                            max_layer_idx = max(max_layer_idx, idx)
                        except:
                            pass
                
                num_layers = max_layer_idx + 1
                
                # Resolve indices
                target_indices = set()
                for idx in config.sharpness_layer_indices:
                    if idx < 0:
                        idx = num_layers + idx
                    target_indices.add(idx)
                
                # Filter
                filtered_mapping = {}
                for friendly_name, actual_name in block_mapping.items():
                    try:
                        parts = friendly_name.split('_')
                        layer_idx = int(parts[1])
                        if layer_idx in target_indices:
                            filtered_mapping[friendly_name] = actual_name
                    except:
                        pass
                
                block_mapping = filtered_mapping
                actual_block_names = list(block_mapping.values())
                print(f"Filtered sharpness calculation to layers: {target_indices}")
            
            # 2. Compute Sharpness (Global + Blocks)
            
            # 2. Compute Sharpness (Global + Blocks)
            
            # Global Results
            global_metrics = {}
            
            # A. Spectral Sharpness (Lanczos)
            if config.compute_spectral_sharpness:
                if config.compute_global_sharpness:
                    res = compute_sharpness(model, loss_wrapper, batch, block_names=None)
                    val = res.get('global', 0.0)
                    global_metrics['global_spectral_sharpness'] = val
                    print(f"Step {step}: Spectral Sharpness (Global) = {val:.4f}")
                
                if config.compute_block_sharpness and actual_block_names:
                    block_results = compute_sharpness(model, loss_wrapper, batch, block_names=actual_block_names)
                    # Store for logging later
                    global_metrics['block_spectral_results'] = block_results

            # B. Batch Sharpness (Gradient Projection)
            if config.compute_batch_sharpness:
                # Import the new function
                from src.metrics import compute_batch_sharpness
                
                if config.compute_global_sharpness:
                    res = compute_batch_sharpness(model, loss_wrapper, batch, block_names=None)
                    val = res.get('global', 0.0)
                    global_metrics['global_batch_sharpness'] = val
                    print(f"Step {step}: Batch Sharpness (Global) = {val:.4f}")
                
                if config.compute_block_sharpness and actual_block_names:
                    block_results = compute_batch_sharpness(model, loss_wrapper, batch, block_names=actual_block_names)
                    global_metrics['block_batch_results'] = block_results
            
            if not config.compute_global_sharpness:
                print(f"Step {step}: EoS evaluation complete (Global skipped)")
            
            # Log metrics
            with torch.no_grad():
                current_loss = model(**batch).loss.item()
                
            metrics = {
                "step": step,
                "loss": current_loss,
                "lr": config.learning_rate
            }
            
            # Add Global Metrics
            if 'global_spectral_sharpness' in global_metrics:
                val = global_metrics['global_spectral_sharpness']
                metrics["global_sharpness"] = val # Keep legacy name for compatibility
                metrics["global_spectral_sharpness"] = val
                metrics["global_sharpprod"] = val * config.learning_rate
                
            if 'global_batch_sharpness' in global_metrics:
                val = global_metrics['global_batch_sharpness']
                metrics["global_batch_sharpness"] = val
                metrics["global_batch_sharpprod"] = val * config.learning_rate
            
            # Add block metrics using friendly names
            if 'block_spectral_results' in global_metrics:
                block_results = global_metrics['block_spectral_results']
                for friendly_name, actual_name in block_mapping.items():
                    if actual_name in block_results:
                        val = block_results[actual_name]
                        metrics[f"{friendly_name}_sharpness"] = val
                        metrics[f"{friendly_name}_sharpprod"] = val * config.learning_rate

            if 'block_batch_results' in global_metrics:
                block_results = global_metrics['block_batch_results']
                for friendly_name, actual_name in block_mapping.items():
                    if actual_name in block_results:
                        val = block_results[actual_name]
                        metrics[f"{friendly_name}_batch_sharpness"] = val
                        metrics[f"{friendly_name}_batch_sharpprod"] = val * config.learning_rate
                
            logger.log(metrics)
            
        # --- Optimization Step ---
        outputs = model(**batch)
        loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        micro_step += 1
        
        # Perform optimizer step only after accumulating enough gradients
        if micro_step % config.gradient_accumulation_steps == 0:
            # Gradient Clipping
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            # Log the actual loss (unscaled) for the user
            progress_bar.set_postfix(loss=loss.item() * config.gradient_accumulation_steps)
            step += 1
            
            # Save Checkpoint
            if step % config.save_interval == 0:
                save_path = output_dir / f"checkpoint-{step}"
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
            
    # Save Final Model
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Saved to {final_path}")
    
    # Generate Plots
    # print("Generating EoS plots...")
    # plot_eos_metrics(
    #     csv_path=output_dir / "metrics.jsonl", # Updated to jsonl
    #     output_dir=output_dir,
    #     learning_rate=config.learning_rate
    # )
    
    logger.finalize()
    print(f"Experiment {config.experiment_name} complete!")

def main():
    parser = argparse.ArgumentParser(description="Run SFT Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Expand Config
    configs = expand_config(args.config)
    print(f"Found {len(configs)} experiment configurations.")
    
    for i, config in enumerate(configs):
        print(f"\n=== Running Experiment {i+1}/{len(configs)} ===")
        train(config)

if __name__ == "__main__":
    main()
