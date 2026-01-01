from dataclasses import field

import torch

from Senti.modules.dataclasses.base import tensor_node


@tensor_node
class ActionPolicy:
    a: torch.Tensor
    pi: torch.Tensor
    allowed_keys: list[str] = field(default_factory=lambda: ['a', 'pi'])

    @property
    def a_pi(self):
        """loopback attribute to return the whole object"""
        return self

    def __post_init__(self):
        assert isinstance(self.a, torch.Tensor)
        assert isinstance(self.pi, torch.Tensor)