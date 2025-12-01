import copy
from typing import Generic, List, Type, TypeVar

import torch
from base import BaseState
from torch import nn

# java has formally brainrotted my stupid little head
# i just love generics so much
T = TypeVar("T", bound="BaseState")


class DepthState(Generic[T], BaseState):
    def __init__(
        self,
        depth: int,
        values: List[T],
        stateclass: Type[T],
    ):
        """
        depth: extra arg to sanity check
        values: list of values, depth length, of T
        stateclass: T extends BaseState
        of course python isnt strictly typed, but if you are reading this
        you cant blame me for not trying to be clear
        """
        super().__init__()
        cls = self.__class__.__name__
        assert len(values) == depth, f"{cls}: Assertion failed. Passed in list of values not equal to" \
            f"depth. Depth is {depth} len values is {len(values)}"
        self.depth = depth
        self.values = values
        self.stateclass = stateclass

    def clone(self, detach=False, freeze=False) -> "DepthState":
        new_node = copy.deepcopy(self)
        new_node.values = [v.clone(detach, freeze) for v in self.values]

        return new_node
