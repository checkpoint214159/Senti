import torch
from torch.func import vjp, vmap
from torch.utils._pytree import tree_flatten


# DEPRECIATED:: VERY USELESS
class VMapBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        func, in_dims, params_dict, randomness, out_spec_container, *args):
        # print('gradients enabled before vmap:', [p.requires_grad for p in params_dict.values()])
        vmapped_fn = vmap(func, in_dims=in_dims, randomness=randomness)
        # print('gradients enabled before vjp:', [p.requires_grad for p in params_dict.values()])
        
        out, pullback = vjp(vmapped_fn, params_dict, *args)
        # print('gradients enabled after vjp:', [p.requires_grad for p in params_dict.values()])
        # out is a pytree object. for our custom types we can / should
        # do a flattening to be compatible with legacy autograd code.
        if isinstance(out, torch.Tensor):
            print('out in vmap bridge has grad?', out.grad_fn, out.requires_grad)
        # print('pullback in vmap bridge?', pullback)
        flat_outputs, out_spec = tree_flatten(out)
        # print('flat_outputs in vmap bridge?', flat_outputs)
        print('flat_outputs have grad?', [f.grad_fn for f in flat_outputs])
        print('out_spec in vmap bridge?', out_spec)
        
        ctx.pullback = pullback
        ctx.out_spec = out_spec
        out_spec_container.append(out_spec)
        
        return tuple(flat_outputs)

    @staticmethod
    def backward(ctx, grad_output):
        all_grads = ctx.pullback(grad_output)
        print('all_grads?', all_grads)
        
        return (
            None,          # func
            None,          # in_dims
            all_grads[0],  # params_dict (the dict of grads)
            None,          # randomness
            None,          # out_spec_container
            *all_grads[1:] # the rest of the args (obs_sequence, etc)
        )
