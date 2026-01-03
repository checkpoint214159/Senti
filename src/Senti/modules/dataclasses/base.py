import copy
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

import torch
import torch.nn as nn


def _apply_node_logic(cls):
    """Core logic shared by all keyable types"""
    cls = dataclass(cls, init=False, eq=False)
    
    # i prefer explicit string methods, as I think for semantics it makes it clearer
    # what is being accessed, improving readability
    cls.get = lambda self, key: getattr(self, key)
    cls.set = lambda self, key, val: setattr(self, key, val)
    cls.has = lambda self, key: hasattr(self, key)
    cls.keys = lambda self: [f.name for f in fields(self)]
    
    if not hasattr(cls, 's'):
        cls.s = property(lambda self: self)
    return cls

# wrapper to allow for dataclass logic whilst calling nn.Module init first.
# this is technically dangerous, as we are using dataclass but not treating it
# as actaul comparable data. This means we simply disable the auto-__eq__ generation,
# telling python we arent really treating this as comparable data, instead a container for data,
# or sort of a schema for data.
def state_node(cls):
    cls = _apply_node_logic(cls)
    
    def __init__(self, *args, **kwargs):
        super(cls, self).__init__()
        
        cls_fields = fields(cls)
        
        for i, val in enumerate(args):
            setattr(self, cls_fields[i].name, val)
            
        for name, val in kwargs.items():
            setattr(self, name, val)
            
        for field in cls_fields:
            if not hasattr(self, field.name):
                if field.default_factory is not None:
                    setattr(self, field.name, field.default_factory())
                else:
                    setattr(self, field.name, field.default)

        if hasattr(self, '__post_init__'):
            self.__post_init__()

    cls.__init__ = __init__
    return cls

def tensor_node(cls):
    """extend to include tensor-only containers."""
    cls = state_node(cls)
    
    def detach(self):
        """Returns a new instance with all fields detached from the graph"""
        new_data = {f.name: getattr(self, f.name).detach() 
                    if hasattr(getattr(self, f.name), 'detach') 
                    else getattr(self, f.name) 
                    for f in fields(self)}

        return type(self)(**new_data)

    def clone(self):
        """Deep clone of the tensors in the node"""
        new_data = {f.name: getattr(self, f.name).clone() 
                    if hasattr(getattr(self, f.name), 'clone') 
                    else getattr(self, f.name) 
                    for f in fields(self)}
        return type(self)(**new_data)

    cls.detach = detach
    cls.clone = clone
    return cls


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
