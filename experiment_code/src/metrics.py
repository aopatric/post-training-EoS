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
        # Check if vector has NaNs (SciPy might pass them if previous steps failed)
        # But here we just run eigsh.
        
        evals, _ = eigsh(operator, k=neigs, which='LA', tol=1e-6, maxiter=10000)
        # Return the largest algebraic eigenvalue
        return float(np.max(evals))
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
            # 4. Compute HVP (H * grad)
            # grad(grad_norm_sq) = 2 * H * grad? No.
            # grad(grad^T * grad) = 2 * grad^T * H?
            # Let's stick to the definition: HVP(v) = grad(grad(L)^T * v)
            # Here v = grad(L). So dot = grad(L)^T * grad(L) = grad_norm_sq.
            # So grad(dot) = grad(grad_norm_sq) = 2 * H * grad(L).
            # Wait, d/dtheta (g^T g) = 2 g^T (dg/dtheta) = 2 g^T H.
            # So grad(grad_norm_sq) gives 2 * H * g.
            # So we need to divide by 2?
            # Let's verify Pearlmutter trick.
            # dot = grad * v.
            # grad(dot) = H * v.
            # Here v is NOT constant, v is grad(theta).
            # So if we differentiate grad^T * grad, we get term for first grad and second grad.
            # It's 2 * H * grad.
            # So yes, we need to divide by 2 if we differentiate grad_norm_sq directly.
            
            # However, in compute_hvp, 'vector' is treated as constant (detached).
            # If we pass flat_grads as a detached vector, then we are good.
            # But here I am differentiating grad_norm_sq which depends on theta twice.
            
            # Let's do it safely using the standard HVP trick with detached vector.
            v = flat_grads.detach()
            
            # Re-compute dot with detached vector to treat it as constant for the second derivative
            # dot = grad^T * v
            dot = flat_grads.mul(v).sum()
            
            # grad(dot) = H * v
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
    
    # Prepare for block-wise if requested
    # We will only support global for now in this pass to ensure correctness of the complex metric
    # But we can allow 'global' key in results.
    
    for i in range(mc_samples):
        try:
            # Get the next batch
            batch = next(data_iterator)
        except StopIteration:
            break

        # Move batch to device (dataloader usually does this, but run_sft manual loop does it)
        # Inside run_sft, the dataloader yields cpu tensors usually, and we move them.
        # But here we are passed the dataloader directly.
        # We need to ensure batch is on device.
        if isinstance(batch, dict):
             batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
             batch = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]

        # --- Phase 1: Calculate the Mini-Batch Gradient (g_xi) ---
        
        # Zero previous gradients
        model.zero_grad()
        
        # Forward pass and loss calculation
        # The loss_fn wrapper handles the forward pass: loss_fn(model, batch) -> tensor
        loss = loss_fn(model, batch)

        # First backward pass computes dL/d(params) = g_xi
        # We use create_graph=True to allow for the second derivative calculation (HvP)
        # We need to manually call grad() because loss.backward() accumulates in .grad which is harder to HVP
        
        gradients = grad(loss, params, create_graph=True, retain_graph=True)

        # --- Phase 2: Calculate Denominator (||g_xi||^2) ---
        # Component: ||g_xi||^2
        
        # Optimization: Flatten gradients once
        flat_grads = parameters_to_vector(gradients)
        grad_norm_sq = flat_grads.pow(2).sum()
        denominator_sum += grad_norm_sq.item()

        # --- Phase 3: Calculate Numerator (g_xi^T * H * g_xi) via HvP ---
        # The quantity g_xi^T * H * g_xi is the second directional derivative (HvP) 
        # of the loss function L, evaluated along the direction of the gradient g_xi.

        # Calculate the Hessian-Vector Product (H * g_xi) where the vector is g_xi itself.
        # We need the HvP: H * g_xi
        
        # In the user snippet:
        # hv_product = grad(gradients, params, grad_outputs=gradients, retain_graph=True)
        # That syntax suggests gradient of gradients?
        # torch.autograd.grad(outputs, inputs, grad_outputs=...) 
        # If outputs is a list of tensors (gradients), we need grad_outputs.
        
        # User snippet:
        # hv_product = grad(gradients, params, grad_outputs=gradients, retain_graph=True)
        # This computes sum(gradients * grad_outputs) gradients?
        # d/d_params ( sum(gradients_i * gradients_i) ) ? 
        # Yes, if you pass grad_outputs=gradients, it computes grad( sum(gradients * gradients) ).
        # Wait, standard Jacobian-vector product.
        # if y = gradients(x). We want J * v. 
        # torch.autograd.grad(y, x, grad_outputs=v) computes v^T * J.
        # Since Hessian is symmetric, J = H. v^T H = (H v)^T.
        # So yes, this gives H * g_xi.
        
        hv_product = grad(gradients, params, grad_outputs=gradients, retain_graph=True)

        # Component: g_xi^T * (H * g_xi)
        # This is the dot product of the original gradient g_xi and the HvP (H * g_xi)
        
        # User snippet:
        # g_T_H_g = sum(torch.sum(g_i * hv_i) for g_i, hv_i in zip(gradients, hv_product))
        
        # We can use flattened versions
        flat_hv = parameters_to_vector(hv_product)
        
        # We used flat_grads for detached vector for dot product
        # Ensure we use the proper graph-connected variables if needed?
        # The user snippet sums them up.
        # numerator_sum += g_T_H_g.detach()
        # We only need the scalar value for the accumulation.
        
        g_T_H_g = flat_grads.mul(flat_hv).sum()
        numerator_sum += g_T_H_g.item()
        
        # Clean up graph
        # del gradients, hv_product, loss
        # params are kept
        
    # 4. Final IAS Calculation
    if denominator_sum == 0:
        results['global'] = 0.0
    else:
        results['global'] = numerator_sum / denominator_sum
    
    # Restore model mode
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