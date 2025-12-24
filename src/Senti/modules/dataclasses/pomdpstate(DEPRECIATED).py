"""
Class that implements state as per (my own interpretation of) the POMDP formulation.
"State" is split into two main parts, h = recurrent hidden state that represents "some formulation of the world",
and z = latent state representative of "some formulation of what the agent observes/senses" (that isnt recurrent,
or isnt treated as it and passed into our recurrent models)

oh yeah we also deal with depth here
"""

import copy
from typing import Generic, List, Type, TypeVar

import torch
from torch import nn

from Senti.modules.dataclasses.base import BaseState
from Senti.modules.dataclasses.normal import DepthNormal, Normal
from Senti.modules.dataclasses.recurrentkvstate import RecurrentKVState


class POMDPState(BaseState):
    """
    Acts as a container of various states.
    """
    def __init__(self,
        h: DepthNormal[RecurrentKVState], z: Normal, a: Normal):
        super().__init__()
        assert isinstance(h, DepthNormal)
        assert isinstance(z, Normal)
        assert isinstance(a, Normal)
        
        self.h = h
        self.z = z
        self.a = a
        self.allowed_keys = ['h', 'z', 'a']
    
    def mean(self, key: str):
        """returns mean of the key specified by user"""
        assert key in self.allowed_keys, \
            f'Assertion failed. Key given to POMDPState.mean is in allowed keys {self.allowed_keys}'
        return getattr(self, key).mean
    
    def log_std(self, key: str):
        """returns log_std of the key specified by user"""
        assert key in self.allowed_keys, \
            f'Assertion failed. Key given to POMDPState.log_std is in allowed keys {self.allowed_keys}'
        return getattr(self, key).log_std
    
    def as_distribution(self, key: str):
        """returns as_distribution of the key specified by user"""
        assert key in self.allowed_keys, \
            f'Assertion failed. Key given to POMDPState.as_distribution is in allowed keys {self.allowed_keys}'
        return getattr(self, key).as_distribution
    
    def get(self, key: str):
        assert key in self.allowed_keys, f"Invalid key {key}"
        return getattr(self, key)

    def set(self, key: str, value):
        assert key in self.allowed_keys, f"Invalid key {key}"
        setattr(self, key, value)

    def clone(self, detach=False, freeze=False) -> "POMDPState":
        new = POMDPState(
            h=self.h.clone(detach, freeze),
            z=self.z.clone(detach, freeze),
            a=self.a.clone(detach, freeze),
        )
        new.allowed_keys = self.allowed_keys[:]  # shallow copy safe
        return new
    
    def all_means(self) -> tuple:
        """prevent the user from verbosely calling all mean"""
        return self.mean('h'), self.mean('z'), self.mean('a')
    
    @classmethod
    def _from_raw_h(
        cls,
        h_state: list[tuple[torch.Tensor]],
        z_value: torch.Tensor,
        a_dim: int,
    ) -> "POMDPState":
        """
        nmmo specific init method, to safely bridge between the structured
        constructor and the chaos of the agent
        TODO this was a quick patch and isnt well made but we will see later
        """
        h = DepthNormal(
            depth=len(h_state),
            values=[RecurrentKVState(h[0], h[1], init_log_std=0.5)
                for h in h_state],
            stateclass=RecurrentKVState
        )
        z = Normal(z_mean=z_value, z_log_std_shape=z_value.shape)
        a = Normal(z_mean_shape=(a_dim,), z_log_std_shape=(a_dim,))
        return POMDPState(h, z, a)

    @classmethod
    def _from_dataclass_h(
        cls,
        h: DepthNormal,
        z_value: torch.Tensor,
        a_dim: int,
    ) -> "POMDPState":
        """
        TODO this was a quick lazy func and isnt well made but we will see later
        """
        z = Normal(z_mean=z_value, z_log_std_shape=z_value.shape)
        a = Normal(z_mean_shape=(a_dim,), z_log_std_shape=(a_dim,))
        return POMDPState(h, z, a)
    
    def raw(self, key):
        assert key in self.allowed_keys, \
            f'Assertion failed. Key given to POMDPState.mean is in allowed keys {self.allowed_keys}'
        return getattr(self, key).raw()


# TODO write tests here 
