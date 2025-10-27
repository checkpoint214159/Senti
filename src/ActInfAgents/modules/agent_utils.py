import copy
from collections import defaultdict
from typing import Any, Type

import torch

from ActInfAgents.modules.state import StateNode


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


class TensorCache(Cache):
    """
    Narrower typing, only allowing for Tensor type caches
    """
    def __init__(self):
        super().__init__(torch.Tensor)

    def clone(self, detach=False):
        """
        Creates a clone, optionally with detach kwarg.
        """
        self._data = {k: v.clone() for k, v in self._data.items()}
        if detach:
            self._data = {k: v.detach() for k, v in self._data.items()}
        return self


class StateCache(Cache):
    """
    A cache specifically for StateNode,
    with additional ops.
    
    1. Cloning
    2. Detaching
    3. Freezing
    4. Replicating
    """

    def __init__(self):
        super().__init__(
            value_type=StateNode,
            key_type=int,    
        )

    def replica(self, detach=False, freeze=False) -> "StateCache":
        """
        Creates a replica of itself using clone, optionally with detach and freeze kwargs.
        Because StateNode will probably have a specific cloning method, this replica method overrides that of Cache.
        """
        cloned_cache = copy.deepcopy(self)
        cloned_cache._data = {k: v.clone(detach, freeze) for k, v in cloned_cache._data.items()}
        return cloned_cache
    
    def get_timesteps(self, timesteps: list) -> list:
        """
        Helper method to get only certain timesteps from the cache and return them with formatting.
        Calls s.get_h_states() from StateNode

        Args:
            - timesteps: list of timesteps to get from StatesCache.
        """
        h_states = []
        for t in timesteps:
            s = self.get(t)
            h_state = s.get_h_states()
            h_states.append(h_state)

        return h_states
    
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