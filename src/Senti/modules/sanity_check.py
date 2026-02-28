import torch
from torch.func import vmap, vjp
from torch.utils._pytree import tree_flatten, tree_unflatten

# 1. SETUP: Hardcoded Weights and Data
# We use a dict to mimic your 'merged_params'
params = {
    "weight": torch.randn(8, 4, requires_grad=True), # 8 agents, 4-dim input
    "bias": torch.zeros(8, 1, requires_grad=True)
}

# 16 agents, 4 features each
obs_batch = torch.randn(16, 4) 

# 2. THE FUNCTIONAL CORE
def f_predict(p, x):
    """
    Standard linear op: y = wx + b
    Note: p is a 'Dual Tensor' here (no .grad inside this scope)
    """
    return torch.matmul(p["weight"], x.unsqueeze(-1)) + p["bias"]

# 3. APPLYING TRANSFORMS
# vmap: Map over the 16 agents (dim 0 of obs_batch)
# We map params (dim 0) and obs (dim 0)
vmapped_predict = vmap(f_predict, in_dims=(0, 0))

# vjp: Setup the gradient calculation relative to 'params'
# We pass params as the first argument to vjp because that's what we want grads for
output, pullback = vjp(vmapped_predict, params, obs_batch)

# 4. CALCULATE LOSS (Outer Scope)
# At this point, 'output' is a raw tensor. It has NO grad_fn!
loss = output.sum()
print(f"Output shape: {output.shape}") # [16, 8, 1]
print(f"Output has grad_fn? {output.grad_fn}") # None

# 5. THE PULLBACK (The Re-attachment)
# Because 'output' has no grad_fn, we can't do loss.backward().
# Instead, we manually 'pull' the gradients back.
# We create a 'ones' tensor to represent dLoss/dOutput
grad_v = torch.ones_like(output)

# pullback returns a tuple of gradients for every argument passed to vjp
grads_dict, _ = pullback(grad_v)

print("-" * 30)
print("GRADIENT RESULTS:")
print(f"Weight grads exist? {grads_dict['weight'] is not None}")
print(f"Weight grads shape: {grads_dict['weight'].shape}") 
# Matches params['weight'].shape!