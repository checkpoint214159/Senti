import copy
from collections import defaultdict
from typing import Any, Type

import torch
from mixin import StateMixin, TensorMixin, TimestepMixin

from Senti.modules.dataclasses.base import BaseState


class Cache:
    """
    A typed dictionary-like container that enforces all values share the same type.

    This is useful as a base for specialized caches (e.g., StateCache) while still
    behaving like a dict with extra utilities.
    """
    def __init__(self, value_type: Type[Any], key_type=object):
        self.key_type = key_type
        self.value_type = value_type
        self._data = {}

    def add(self, key: Any, value: Any):
        assert isinstance(key, self.key_type), \
            f"Expected key of type {self.key_type}, got {type(key)}"
        assert isinstance(value, self.value_type), \
            f"Expected value of type {self.value_type}, got {type(value)}"
        if key in self._data:
            raise KeyError(f"Key {key} already exists in {self._data}")
        self._data[key] = value

    def get(self, key: Any) -> Any:
        if key not in self._data:
            raise KeyError(f"Key {key} not found in {self._data}")
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def remove(self, key: Any):
        if key not in self._data:
            raise KeyError(f"Key {key} not found in {self.name}")
        del self._data[key]

    def has(self, key: Any) -> bool:
        return key in self._data

    def clear(self):
        self._data.clear()

    def purge_missing(self, list_of_keys):
        """
        For each key in list_of_keys, if it does not exist in the list_of_keys
        but DOES in our cache, remove it.
        """
        keys = list(self._data.keys())
        for key in keys:
            if key not in list_of_keys:
                self.remove(key)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, type={self.value_type.__name__}, size={len(self)})"


class TensorCache(TensorMixin, Cache):
    """
    Narrower typing, only allowing for Tensor type caches
    """
    def __init__(self, **kwargs):
        super().__init__(
            value_type=torch.Tensor,
             **kwargs)


class StateCache(StateMixin, Cache):
    """
    A cache specifically for BaseState and those that inherit from it,
    mostly to easily call replica to each element inside it
    """

    def __init__(self, state_type: BaseState, **kwargs):
        super().__init__(
            value_type=state_type,   
            **kwargs
        )


class StateTimestepCache(StateMixin, StateCache):

    def __init__(self, state_type: BaseState):
        super().__init__(
            value_type=state_type,
            key_type=int,    
        )

    
    def get_all_h_states(self):
        """
        Calls all get_h_states() from all states and all keys,
        and compiles them along the timestep dimensions.

        TODO maybe generalize this to compile only along selected timesteps?
        but convenience first lol
        """
        depthwise_all_s = defaultdict(list) # {int: list[tuple]}
        depthwise_stacked_result = {} # {int: tuple}

        for s in self.values():
            for depth, s_tuple in enumerate(s.get_h_states()):
                depthwise_all_s[depth].append(s_tuple)
        
        # for each depth, stack tuple-element-wise (stack index 1 of tuple1 with index 2 of tuple2)
        for depth, s_tuples in depthwise_all_s.items():
            length_of_tuple = len(s_tuples[0])
            depthwise_stacked_result[depth] = tuple(
                [torch.concat(
                    [s_tuple[i] for s_tuple in s_tuples]
                ) for i in range(length_of_tuple)]
            )

        return depthwise_stacked_result