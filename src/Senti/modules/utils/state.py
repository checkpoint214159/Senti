import copy

import torch
import torch.nn as nn

"""
NOTE: Development of this state / StateNode class was tiresome, as I had to find a way to
make some nice flexible class to contain state of arbitrary shape and complexity, that is also
learnable / parameterizable. This reduces the annoyance of dealing with custom data types
like nn.Parameters in nested lists, dict with at arbitrary keys, etc etc. which I'm facing
when re-purposing the transformer module (from openAI? idk whoever did VPT for minecraft)
"""

class RecurrentMemory(nn.Module):
    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        super().__init__()
        self.key = nn.Parameter(key)
        self.value = nn.Parameter(value)

    def forward(self):
        return self.key, self.value

class StateNode(nn.Module):
    def __init__(
        self,
        h_state: list[tuple[torch.Tensor]],
        z_value: torch.Tensor | None = None,
        a_value: torch.Tensor | None = None,
        z_dim: int | None = None,
        a_dim: int | None = None,
        init_std=1.0,
        min_std=1e-3,
        learn_std=False,
        device=None,
        batch_size=None,
    ):
        """
        Class encapsulating (z_t, h_t) state components:
        - z_t: observation-linked state (can be discrete or continuous)
        - h_t: recurrent memory state (causal_mask, (attention_mask, h state values))
        Both of these are treated as learned parameters. 
        Latent actions meanwhile, are not learned parameters,
        and just saved here for convenience. 

        Args:
            z_value: Init value for z_t
            h_state: Init values for h_t. Note that h is meant to be represented (right now) as 
                (state_mask, h_state), where h_state in theory could be a list of any length Tensors representing the 'idea of state'.
                As of 26/7/25, h_state = (h_keys, h_values), which are keys and values fed to a transformer in our world model.
            a_value: Init value for a_t
            z_dim: Dimension of z_t, if no init value is given
            a_dim: Dimension of a_t, if no init value is given
            init_std: Initial std for continuous z_t and h_t
            min_std: Min std clamp for numerical stability
            learn_std: Whether std of z_t and h_t is learnable
            device: Torch device
            bs: int = 1, default batch size. Must be present if z_dim is chosen
        """
        super().__init__()
        # TODO still uncertain but fairly reasonable. zeroth state should have no observation z
        assert (z_value is not None or z_dim is not None), \
            'Must at least provide one of z_value OR z_dim.'
        assert not (z_value is not None and z_dim is not None), \
            'Must only provide one z_value or z_dim, not both!'
        assert (a_value is not None or a_dim is not None), \
                'Must provide h_value or h_dim.'
        if z_dim is not None:
            assert (batch_size is not None), \
                    'Must provide batch_size if you provide z_dim!'
        self.batch_size = z_value.shape[-1] if z_value is not None else batch_size
        self.state_depth = len(h_state)
        self.min_std = min_std
        self.device = device

        # ---- z_t (observation-affected state) ----
        if z_value is not None:
            self.z_dim = z_value.shape[-1]
            z_init = z_value
        elif z_dim is not None:
            self.z_dim = z_dim
            z_init = torch.zeros(self.batch_size, self.z_dim, device=device)
        else:
            raise AssertionError('How did you get here? Somehow the earlier assertion to check \
                for either z_value or z_dim is present did not work properly. Sue benjamin goh en yang for bad code')
        self.z_mean = nn.Parameter(z_init)

        # ---- h_t (recurrent state, now structured as list of (mask, xf_state)) ----
        self.h_modules = nn.ModuleList()
        self.h_dim = 0  # Optional: can be used for metadata
        for xf_state in h_state:
            h_key, h_val = xf_state
            mem_module = RecurrentMemory(h_key, h_val)
            self.h_modules.append(mem_module)

            self.h_dim += h_key.numel() + h_val.numel()

        # ---- a_t (latent action in some higher level action space) ----
        self.a_dim = a_value.shape[-1] if a_value is not None else a_dim
        self.a = a_value

        if learn_std:  # assume no for debugging now
            self.z_log_std = nn.Parameter(torch.ones(self.z_dim, device=device) * torch.log(torch.tensor(init_std)))
            self.h_log_std = nn.Parameter(torch.ones(self.h_dim, device=device) * torch.log(torch.tensor(init_std)))
        else:
            self.register_buffer("z_log_std", torch.ones(self.z_dim, device=device) * torch.log(torch.tensor(init_std)))
            # TODO update std for h?
            self.register_buffer("h_log_std", torch.ones(self.h_dim, device=device) * torch.log(torch.tensor(init_std)))

    def clone(self, detach=False, freeze=False): # type: ignore
        """
        Scuffed pytorch func to clone parameters.
        This is to provide an interface that we can call as if StateNodes were plain tensors.
        Because detach and freeze ops usually follow with cloning in our use case, for now they are optional args
        we can pass in.
        """
        new_node = copy.deepcopy(self)
        
        # Go through parameters and clone/detach/freeze as needed
        for name, param in new_node.named_parameters():
            new_param = param.clone()
            if detach:
                new_param = new_param.detach()
            new_param.requires_grad_(not freeze)

            # traverse tree to get to the object in which we set. assume no cancerous names e.g h_state.0, because that would break
            # python anyway and you cant have a .0 attribute
            split_name = name.split('.')
            mod = new_node
            for idx, part in enumerate(split_name):
                if idx == len(split_name) - 1:
                    setattr(mod, part, nn.Parameter(new_param))
                if part.isdigit():  # e.g for h_state.0.key, if we are at 0, fail gracefully if h_state is not a ModuleList
                    assert isinstance(mod, nn.ModuleList)
                    mod = mod[int(part)]
                else:
                    mod = getattr(mod, part)
        
        return new_node
    
    def detach(self):
        [param.detach() for _, param in self.named_parameters()]

        return self

    def freeze(self):
        [param.requires_grad_(False) for _, param in self.named_parameters()]

        return self
    
    @property
    def flattened_h_states(self):
        """
        Property to retrieve a flattened form of h_states.
        Right now hardcoded to having keys and values under a module, and a list of these form h.
        TODO generalize this to arbitrary representations of state: i.e more than or less than keys
        Retrieves as follows: [depth_0_keys, depth_0_values, depth_1_keys, depth_1_values, ...] of N length
        After extending the list, then stack it according to B,N,... (preserve batch as first dim)
        Right now both keys and values are of shape (1, 1, hidden_dim), we cat along a new dimension at zero
        """
        h_states = []
        for mod in self.h_modules:
            h_states.extend([mod.key, mod.value])
        # TODO HI IM HERE TO REMIND YOU TO CHECK IF THIS GENERALIZES TO MULTIPLE DEPTHS
        h_states = torch.stack(h_states, 1)  # preserve batch as first dim

        return h_states
    
    @classmethod
    def flatten(self, h_states):
        """
        Helper func to do what the property flattened_h_states does, except
        to an incoming h_state not necessarily tied to us.
        h_states: [(h_keys, h_values), (h_keys, h_values), ...] -> into:
        """
        h_states = [x for tpl in h_states for x in tpl]
        h_states = torch.stack(h_states)
        # print('HI IM HERE TO REMIND YOU TO CHECK IF THIS GENERALIZES TO MULTIPLE DEPTHS')
        return h_states
    
    @classmethod
    def unpack_mask_states(self, h):
        """
        Helper function to unpack the return of the WorldModel, that is [(state_mask, h_states), (state_mask, h_states), ...]
        Will return state_mask and h_states respectively.
        Yes this is redundant but here for simplicity in reading.

        h: list of [(state_mask, h_states), ...], length = depth of state
        Return: (state_mask, ...), (h_states, ...)
        Note if h is of one length, the return WILL be a one size tuple, NOT a normal python thing
        Hardcoded to tuple of two size, cuz zip(*) has some funny (rude) behaviour
        """
        state_masks, h_states = [], []
        for depth in h:
            state_mask, h_state = depth
            state_masks.append(state_mask)
            h_states.append(h_state)
        return state_masks, h_states
    
    @classmethod
    def unpack_flattened_h(self, h_states):
        """
        Helper function to unpack an incoming flattened h_states.
        Expects (2 x Depth, B, 1, Hidden) shape, because we hardcode to key, value hidden states for now
        """
        h_states = list(zip(h_states[::2], h_states[1::2]))
        return h_states 

    def get_h_states(self):
        """
        Helper method to retrieve only h_states w/o masks, i.e [(h_key, h_value), ...]
        Basically the precise formatting required to feed into the WorldModel, minus state masks.
        """
        h = [(mod.key, mod.value) for mod in self.h_modules]
        print('state shapes?', [(mod.key.shape, mod.value.shape) for mod in self.h_modules])

        return h
    
    def get_final_h_value(self):
        """
        Helper func to retrieve only the last h_value from the h module. Right now used to predict action
        """
        final_h_val = self.h_modules[-1].value
        return final_h_val
    
    def get_params(self):
        all_params = list(self.h_modules.parameters())
        all_params.append(self.z_mean)
        return all_params