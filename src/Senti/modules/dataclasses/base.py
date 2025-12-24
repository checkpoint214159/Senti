import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseState(nn.Module, ABC):
    """
    mother of all state. all hail base state.

    It is useful to have a State class as a layer of abstraction, as state itself should be capable
    of flexibly designed, without having the user carefully unpack the data within
    for use.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def assert_type(cls, other):
        assert isinstance(other, cls), (
            f"{cls.__name__}: Assertion failed. Expected {cls.__name__}, "
            f"got {type(other).__name__}"
        )

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
            self._set_nested_attr(new_node, name, nn.Parameter(new_param))

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
    
    def compute_energy(self, other, loss_func):
        """
        for computing energy. the other three arguments are placeholder
        """
        raise NotImplementedError(f"{type(self)} does not implement compute_energy().")
    
    def raw(self):
        """
        entirely for describing the self as some kind of 'raw' datatype.
        the semantics of what is raw depends on the class, hence the requirement.
        however, it can be useful to think of it as some 'default' datatype or form of the data within
        """
        raise NotImplementedError(f"{type(self)} does not implement raw().")

    @staticmethod
    def _set_nested_attr(root: nn.Module, name: str, value):
        """Set an attribute on a nested module path like 'a.b.c'."""
        parts = name.split(".")
        obj = root
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
