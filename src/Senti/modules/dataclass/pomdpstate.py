from dataclasses import field

import torch

from Senti.modules.dataclass.base import BaseState, state_node
from Senti.modules.dataclass.categorical import DepthGroupedCategoricalState
from Senti.modules.dataclass.normal import Normal


@state_node
class POMDPState(BaseState):
    h: DepthGroupedCategoricalState
    z: Normal
    allowed_keys: list[str] = field(default_factory=lambda: ['h', 'z'])

    @property
    def s(self):
        """loopback attribute to return the whole state object"""
        return self

    def __post_init__(self):
        assert isinstance(self.h, DepthGroupedCategoricalState)
        assert isinstance(self.z, Normal)

    def as_distribution(self, key: str):
        return getattr(self, key).as_distribution

    def clone(self, detach=False, freeze=False) -> "POMDPState":
        return POMDPState(
            h=self.h.clone(detach, freeze),
            z=self.z.clone(detach, freeze),
        )

    def get_state(self):
        return torch.concat([self.h.as_tensor(), self.z.sample()])
    
    def __flatten__(self):
        children = (self.h, self.z)
        aux_data = (self.allowed_keys,) 
        return children, aux_data
    
    @classmethod
    def __unflatten__(cls, children, aux_data):
        return cls(
            h=children[0], 
            z=children[1], 
            allowed_keys=aux_data[0],
            as_parameter=False,
        )

    

    