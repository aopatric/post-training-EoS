"""
Module for tracking Edge of Stability metrics, specifically the sharpness (largest Hessian eigenvalue).
Uses manual HVP and Lanczos iteration for robust estimation without external dependencies.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple, Any
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector

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
        with torch.no_grad(): # HVP calculation handles its own graph, but the wrapper shouldn't track
             # Actually HVP NEEDS graph for the double backward. 
             # But the input vector is constant.
             pass
        
        # We need to call hvp_fn which uses autograd.
        # The input 'vec' is from scipy, so it's numpy.
        return hvp_fn(gpu_vec).cpu().numpy()

    operator = LinearOperator((dim, dim), matvec=mv)
    
    # Use 'LA' for Largest Algebraic (since Hessian is real symmetric, eigenvalues are real)
    # For sharpness we want the largest magnitude, but usually it's positive.
    # If we suspect negative curvature dominating, 'LM' (Largest Magnitude) might be better.
    # Reference code uses default which is 'LM' for eigsh? No, eigsh default is 'LM'.
    # Reference code uses `eigsh(operator, neigs)`.
    try:
        evals, _ = eigsh(operator, k=neigs, which='LM', tol=1e-2)
        # Return the largest magnitude eigenvalue
        return float(np.max(np.abs(evals)))
    except Exception as e:
        print(f"Lanczos failed: {e}")
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

