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
            raise KeyError(f"Key {key} already exists in {self.name}")
        self._data[key] = value

    def get(self, key: Any) -> Any:
        if key not in self._data:
            raise KeyError(f"Key {key} not found in {self.name}")
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


class StateCache(Cache):
    """
    A cache specifically for StateNode,
    with additional ops.
    
    1. Cloning
    2. Detaching
    3. Freezing
    4. Replicating
    """

    def __init__(self, name: str = "state_cache"):
        super().__init__(StateNode, name=name)

    def clone(self) -> "StateCache":
        """
        Return a deep copy of this cache with cloned tensors.
        """
        raise NotImplementedError("clone() not yet implemented")

    def detach(self) -> "StateCache":
        """
        Return a copy of this cache where all tensors are detached.
        """
        raise NotImplementedError("detach() not yet implemented")

    def freeze(self) -> "StateCache":
        """
        Return a cache where tensors are frozen (requires_grad=False).
        """
        raise NotImplementedError("freeze() not yet implemented")