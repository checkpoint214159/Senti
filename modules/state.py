import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateNode(nn.Module):
    def __init__(
        self,
        z_value: torch.Tensor | None = None,
        h_value: torch.Tensor | None = None,
        a_value: torch.Tensor | None = None,
        z_dim: int | None = None,
        h_dim: int | None = None,
        a_dim: int | None = None,
        init_std=1.0,
        min_std=1e-3,
        learn_std=False,
        device=None
    ):
        """
        Class encapsulating (z_t, h_t) state components:
        - z_t: observation-linked state (can be discrete or continuous)
        - h_t: recurrent memory state (always continuous)
        Both of these are treated as learned parameters. 
        Latent actions meanwhile, are not learned parameters,
        and just saved here for convenience. 

        Args:
            z_value: Init value for z_t
            h_value: Init value for h_t
            a_value: Init value for a_t
            z_dim: Dimension of z_t, if no init value is given
            h_dim: Dimension of h_t, if no init value is given
            a_dim: Dimension of a_t, if no init value is given
            init_std: Initial std for continuous z_t and h_t
            min_std: Min std clamp for numerical stability
            learn_std: Whether std of z_t and h_t is learnable
            device: Torch device
        """
        super().__init__()
        assert (z_value is not None or z_dim is not None), \
            'Must provide z_value or z_dim.'
        assert (h_value is not None or h_dim is not None), \
            'Must provide h_value or h_dim.'
        assert (a_value is not None or a_dim is not None), \
                'Must provide h_value or h_dim.'

        self.min_std = min_std
        self.device = device

        # ---- z_t (observation-affected state) ----
        self.z_dim = z_value.shape[-1] if z_value is not None else z_dim
        z_init = z_value if z_value is not None else torch.zeros(self.z_dim, device=device)
        self.z_mean = nn.Parameter(z_init)

        # ---- h_t ('world model') ----
        self.h_dim = h_value.shape[-1] if h_value is not None else h_dim
        h_init = h_value if h_value is not None else torch.zeros(self.h_dim, device=device)
        self.h_mean = nn.Parameter(h_init)

        # ---- a_t (latent action in some higher level action space) ----
        self.a_dim = a_value.shape[-1] if a_value is not None else a_dim
        a_init = a_value if a_value is not None else torch.zeros(self.a_dim, device=device)
        self.a = a_init

        if learn_std:  # assume no for debugging now
            self.z_log_std = nn.Parameter(torch.ones(self.z_dim, device=device) * torch.log(torch.tensor(init_std)))
            self.h_log_std = nn.Parameter(torch.ones(self.h_dim, device=device) * torch.log(torch.tensor(init_std)))
        else:
            self.register_buffer("z_log_std", torch.ones(self.z_dim, device=device) * torch.log(torch.tensor(init_std)))
            self.register_buffer("h_log_std", torch.ones(self.h_dim, device=device) * torch.log(torch.tensor(init_std)))

    def clone(self, detach: bool = True, freeze: bool = True): # type: ignore
        """
        Scuffed pytorch func to clone parameters.
        """
        new_node = copy.deepcopy(self)
        
        # Go through parameters and clone/detach/freeze as needed
        for name, param in new_node.named_parameters():
            # print('Cloning', name)
            # print('param', param)
            new_param = param.clone()
            if detach:
                new_param = new_param.detach()
            new_param.requires_grad_(not freeze)
            # print('new_param', new_param)
            setattr(new_node, name, nn.Parameter(new_param, requires_grad=not freeze))
        
        return new_node


    @property
    def state(self):
        """Returns the full latent state s_t = concat(z_t, h_t)"""
        print('STATE PROPERTY IS CALLED FROM STATENODE, VERIFY')
        if self.discrete:
            probs = F.softmax(self.logits, dim=-1)
            assert probs.dim() == 2, "Expected z_t logits of shape (z_dim, discrete_buckets)"
            z_t_flat = probs.view(-1)
        else:
            z_t_flat = self.mean

        assert self.h is not None, "h_t must be initialized."
        return torch.cat([z_t_flat, self.h], dim=-1)
