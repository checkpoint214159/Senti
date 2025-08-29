import copy
from typing import Any, Type

import torch

from AutonoMC.modules.state import StateNode


class Cache:
    """
    A typed dictionary-like container that enforces all values share the same type.

    This is useful as a base for specialized caches (e.g., StateCache) while still
    behaving like a dict with extra utilities.
    """
    def __init__(self, value_type: Type[Any]):
        self.value_type = value_type
        self._data: dict[Any, Any] = {}

    def add(self, key: Any, value: Any):
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
        super().__init__(StateNode)

    def replica(self, detach=False, freeze=False) -> "StateCache":
        """
        Creates a replica of itself using clone, optionally with detach and freeze kwargs.
        Because StateNode will probably have a specific cloning method, this replica method overrides that of Cache.
        """
        cloned_cache = copy.deepcopy(self)
        cloned_cache._data = {k: v.clone(detach, freeze) for k, v in cloned_cache._data.items()}
        return cloned_cache
