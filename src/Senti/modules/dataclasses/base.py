import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseState(nn.Module, ABC):
    """
    mother of all state. all hail base state.
    """

    def __init__(self):
        super().__init__()

    def clone(self, detach=False, freeze=False):
        """
        return a cloned version of this state
        unified method of calling clone, as
        """
        new_node = copy.deepcopy(self)
        
        # Go through parameters and clone/detach/freeze as needed
        for name, param in new_node.named_parameters():
            new_param = param.clone()
            if detach:
                new_param = new_param.detach()
            new_param.requires_grad_(not freeze)

            setattr(new_node, name, new_param)

        return new_node
    
    # --------- canon interfaces for loss adaptation -------

    def as_distribution(self):
        """
        Optional: return a torch.distributions.Distribution object.
        override if meaningful.
        """
        raise NotImplementedError(f"{type(self)} does not implement as_distribution().")

    def as_tensor(self):
        """
        Optional: return flat tensor representation (if supported).
        """
        raise NotImplementedError(f"{type(self)} does not implement as_tensor().")

    def sample(self, n=None):
        raise NotImplementedError(f"{type(self)} does not implement sample().")

    def log_prob(self, x):
        """For probabilistic states."""
        raise NotImplementedError(f"{type(self)} does not implement log_prob().")
