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
from base import BaseState
from depthstate import DepthState
from normal import BaseNormal, Normal
from recurrentkvstate import RecurrentKVState
from torch import nn


class POMDPState(BaseState):
    """
    Acts as a container of various states.
    """
    def __init__(self, h: DepthState, z: Normal, a: Normal):
        super().__init__()
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

# TODO write tests here 
