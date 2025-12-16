"""
Module for tracking Edge of Stability metrics, specifically the sharpness (largest Hessian eigenvalue).
Uses manual HVP and Lanczos iteration for robust estimation without external dependencies.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple, Any
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector
from torch.autograd import grad

def compute_hvp(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    batch: Any,
    vector: torch.Tensor
) -> torch.Tensor:
    """
    Compute a Hessian-vector product using the Pearlmutter trick (autograd of autograd).
    """
    # 1. Compute loss
    loss = loss_fn(model, batch)
    
    # 2. Compute gradients w.r.t. parameters
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.zeros_like(vector)
        
    grads = torch.autograd.grad(loss, inputs=params, create_graph=True)
    
    # 3. Compute dot product of gradients and vector
    # vector is a single flattened tensor, we need to reshape it to match params or flatten grads
    # It's easier to flatten grads
    flat_grads = parameters_to_vector(grads)
    
    dot = flat_grads.mul(vector).sum()
    
    # 4. Compute gradient of dot product w.r.t. parameters
    # This gives H*v
    grads_2 = torch.autograd.grad(dot, inputs=params, retain_graph=True)
    hvp = parameters_to_vector(grads_2)
    
    # Check for NaNs in HVP
    if torch.isnan(hvp).any():
        # print("Warning: NaN found in HVP, returning zeros")
        return torch.zeros_like(hvp)
    
    return hvp.detach() # Detach to save memory, we don't need higher order derivatives of this

def lanczos(
    hvp_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    neigs: int = 1,
    device: str = 'cuda'
) -> float:
    """
    Invoke the Lanczos algorithm to compute the leading eigenvalues.
    """
    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float32, device=device)
        return hvp_fn(gpu_vec).cpu().numpy()

    operator = LinearOperator((dim, dim), matvec=mv)
    
    try:
        evals, _ = eigsh(operator, k=neigs, which='LA', tol=1e-6, maxiter=10000)
        return float(np.max(evals))
    except Exception as e:
        return 0.0

def compute_sharpness(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    batch: Any,
    block_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Computes sharpness (top Hessian eigenvalue) using manual HVP + Lanczos.
    
    Args:
        model: The model.
        loss_fn: Function that takes (model, batch) and returns scalar loss.
        batch: Data batch.
        block_names: List of module names to analyze.
        
    Returns:
        Dict[str, float]: Dictionary mapping block names (or 'global') to sharpness values.
    """
    results = {}
    
    # Determine device
    try:
        param = next(model.parameters())
        device = param.device
    except:
        device = 'cpu'

    # Helper for HVP closure
    def get_hvp_fn(current_model):
        def fn(v):
            return compute_hvp(current_model, loss_fn, batch, v)
        return fn

    if block_names is None:
        # Global sharpness
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return {}
            
        num_params = sum(p.numel() for p in params)
        hvp_fn = get_hvp_fn(model)
        
        try:
            val = lanczos(hvp_fn, num_params, device=device)
            results['global'] = val
        except Exception as e:
            print(f"Error computing global sharpness: {e}")
            results['global'] = 0.0
    else:
        # Block-wise sharpness
        # We need to isolate gradients to specific blocks.
        
        # 1. Save current requires_grad state
        grad_state = {p: p.requires_grad for p in model.parameters()}
        
        # 2. Disable all
        for p in model.parameters():
            p.requires_grad = False
            
        # 3. Iterate blocks
        for name, module in model.named_modules():
            if name in block_names:
                # Enable for this block
                # Only enable params that were originally trainable (requires_grad was True in grad_state)
                params = []
                for p in module.parameters():
                    if grad_state.get(p, False): # Check if it was originally trainable
                        p.requires_grad = True
                        params.append(p)
                
                if not params:
                    continue
                
                num_params = sum(p.numel() for p in params)
                
                try:
                    # Compute
                    # We pass the FULL model, but only enabled params will have grad
                    # compute_hvp filters by requires_grad
                    hvp_fn = get_hvp_fn(model)
                    val = lanczos(hvp_fn, num_params, device=device)
                    results[name] = val
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error computing sharpness for {name}: {e}")
                    results[name] = 0.0
                
                # Disable again for next block
                for p in params:
                    p.requires_grad = False
                    
        # 4. Restore state
        for p, state in grad_state.items():
            p.requires_grad = state
            
    return results

def compute_batch_sharpness(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    batch: Any,
    block_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Computes 'Batch Sharpness' (directional curvature along the gradient).
    Formula: (grad^T * H * grad) / (grad^T * grad)
    
    Args:
        model: The model.
        loss_fn: Function that takes (model, batch) and returns scalar loss.
        batch: Data batch.
        block_names: List of module names to analyze.
        
    Returns:
        Dict[str, float]: Dictionary mapping block names (or 'global') to batch sharpness values.
    """
    results = {}
    
    if block_names is None:
        # Global Batch Sharpness
        
        # 1. Compute Loss
        loss = loss_fn(model, batch)
        
        # 2. Compute Gradients
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return {}
            
        grads = torch.autograd.grad(loss, inputs=params, create_graph=True)
        flat_grads = parameters_to_vector(grads)
        
        # 3. Compute Norm Squared
        grad_norm_sq = flat_grads.pow(2).sum()
        
        if grad_norm_sq.item() == 0:
            results['global'] = 0.0
        else:
            v = flat_grads.detach()
            
            dot = flat_grads.mul(v).sum()
            
            hvp = parameters_to_vector(torch.autograd.grad(dot, inputs=params, retain_graph=True))
            
            numerator = v.mul(hvp).sum()
            batch_sharpness = numerator / grad_norm_sq
            results['global'] = batch_sharpness.item()
            
    else:
        # Block-wise Batch Sharpness
        
        # 1. Save state
        grad_state = {p: p.requires_grad for p in model.parameters()}
        
        # 2. Disable all
        for p in model.parameters():
            p.requires_grad = False
            
        # 3. Iterate blocks
        for name, module in model.named_modules():
            if name in block_names:
                # Enable for this block
                params = []
                for p in module.parameters():
                    if grad_state.get(p, False):
                        p.requires_grad = True
                        params.append(p)
                
                if not params:
                    continue
                    
                try:
                    # Compute Loss
                    loss = loss_fn(model, batch)
                    
                    # Compute Gradients
                    grads = torch.autograd.grad(loss, inputs=params, create_graph=True)
                    flat_grads = parameters_to_vector(grads)
                    
                    grad_norm_sq = flat_grads.pow(2).sum()
                    
                    if grad_norm_sq.item() == 0:
                        results[name] = 0.0
                    else:
                        # Compute HVP
                        v = flat_grads.detach()
                        dot = flat_grads.mul(v).sum()
                        hvp = parameters_to_vector(torch.autograd.grad(dot, inputs=params, retain_graph=True))
                        
                        numerator = v.mul(hvp).sum()
                        batch_sharpness = numerator / grad_norm_sq
                        results[name] = batch_sharpness.item()
                        
                except Exception as e:
                    print(f"Error computing batch sharpness for {name}: {e}")
                    results[name] = 0.0
                
                # Disable again
                for p in params:
                    p.requires_grad = False
                    
        # 4. Restore state
        for p, state in grad_state.items():
            p.requires_grad = state
            
    return results

def compute_ias(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    dataloader: Any,
    mc_samples: int = 32,
    block_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Estimates the Interaction-Aware Sharpness (IAS) of the model's current state
    using Monte Carlo sampling and the Hessian-Vector Product (HvP).
    
    Adaptation of the snippet provided by the user to fit the codebase conventions.
    """
    results = {}
    
    # Check dataloader length if possible
    try:
        total_batches = len(dataloader)
        if mc_samples > total_batches:
            print(f"Warning: Requested {mc_samples} samples, but DataLoader only has {total_batches} batches. Using max batches.")
            mc_samples = total_batches
    except:
        pass # Iterable dataset or unknown length

    if mc_samples == 0:
        return results

    # Tensors to accumulate the sums of the numerator and denominator components
    # We will compute GLOBAL IAS
    numerator_sum = 0.0
    denominator_sum = 0.0
    
    # 1. Freeze model and set to evaluation mode
    # Store original training mode
    was_training = model.training
    model.eval()
    
    # 2. Reset the DataLoader iterator to ensure fresh sampling
    data_iterator = iter(dataloader)
    
    # Identify trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        if was_training: model.train()
        return {}
        
    device = params[0].device
    
    for i in range(mc_samples):
        try:
            # Get the next batch
            batch = next(data_iterator)
        except StopIteration:
            break

        if isinstance(batch, dict):
             batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
             batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]

        model.zero_grad()
        
        loss = loss_fn(model, batch)

        gradients = grad(loss, params, create_graph=True, retain_graph=True)

        flat_grads = parameters_to_vector(gradients)
        grad_norm_sq = flat_grads.pow(2).sum()
        denominator_sum += grad_norm_sq.item()
        
        hv_product = grad(gradients, params, grad_outputs=gradients, retain_graph=True)

        flat_hv = parameters_to_vector(hv_product)
        
        g_T_H_g = flat_grads.mul(flat_hv).sum()
        numerator_sum += g_T_H_g.item()
        
    if denominator_sum == 0:
        results['global'] = 0.0
    else:
        results['global'] = numerator_sum / denominator_sum
    
    if was_training:
        model.train()
    
    return results


if __name__ == "__main__":
    print("Metrics module (Manual HVP version)")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\n--- Model Integration Test ---")
        model_name = "EleutherAI/pythia-70m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device)
        
        # Dummy batch
        text = "Hello, how are you?"
        inputs = tokenizer(text, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        def loss_fn(m, b):
            return m(**b).loss
        
        # Run sharpness
        print("Computing sharpness...")
        results = compute_sharpness(model, loss_fn, inputs, block_names=["gpt_neox.layers.0"])
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"Test failed: {e}")