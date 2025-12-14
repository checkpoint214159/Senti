import copy
from collections import defaultdict
from typing import Any, Type

import torch
from torch import nn

from Senti.modules.dataclasses.base import BaseState
from Senti.modules.utils.mixins import StateMixin, TensorMixin, TimestepMixin


class Cache:
    """
    A typed dictionary-like container that enforces all values share the same type.

    This is useful as a base for specialized caches (e.g., StateCache) while still
    behaving like a dict with extra utilities. We love functional programming so we will make it more monadic where available
    """
    def __init__(self, value_type: Type[Any], key_type=object):
        self.key_type = key_type
        self.value_type = value_type
        self._data = {}

    @classmethod
    def of(cls, data:dict, value_type:Type[Any] | None = None) -> "Cache":
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}.of: expected dict, got {type(data).__name__}")

        if len(data) == 0 or value_type is None:
            value_type = object
        else:
            iterator = iter(data.values())
            first_value = next(iterator)
            value_type = type(first_value)
            for v in iterator:
                if not isinstance(v, value_type):
                    raise TypeError(
                        f"{cls.__name__}.of: inconsistent value types: "
                        f"{type(v).__name__} vs {value_type.__name__}"
                    )
        self = cls(value_type=value_type)
        self._data = dict(data)
        return self
    
    def vmap(self, func) -> "Cache":
        """
        god i hate what cs2030 has done to me
        maps v -> func(v) for all v 
        """
        new_data = {k: func(v) for k, v in self._data.items()}
        return type(self).of(new_data)
    
    def kmap(self, func) -> "Cache":
        """
        god i hate what cs2030 has done to me
        maps k -> func(k) for all k 
        """
        new_data = {func(k): v for k, v in self._data.items()}
        return type(self).of(new_data)


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
    
    def get_list(self, keys: list[Any]) -> Any:
        return [self.get(k) for k in keys]

    def keys(self):
        return self._data.keys()

    def values(self):
        return list(self._data.values())

    def items(self):
        return self._data.items()

    def remove(self, key: Any):
        if key not in self._data:
            raise KeyError(f"Key {key} not found in {self.name}")
        del self._data[key]

    def has(self, key: Any) -> bool:
        return key in self._data
    
    def has_list(self, keys: list[Any]) -> Any:
        return all([self.has(k) for k in keys])

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
        cls_name = self.__class__.__name__
        name = getattr(self, "name", cls_name)

        if name == cls_name:
            return f"{cls_name}(type={self.value_type.__name__}, size={len(self)})"

        return f"{cls_name}(name={name}, type={self.value_type.__name__}, size={len(self)})"


class TensorCache(TensorMixin, Cache):
    """
    Narrower typing, only allowing for Tensor type caches
    """
    def __init__(self, **kwargs):
        super().__init__(
            value_type=torch.Tensor,
             **kwargs)
        
    @classmethod
    def of(cls, data:dict):
        return super().of(data, value_type=torch.Tensor)
        
    def clone(self, detach=False):
        """
        Creates a clone, optionally with detach kwarg.
        """
        new = copy.deepcopy(self)
        if detach:
            new._data = {k: v.detach() for k, v in new._data.items()}
        else:
            new._data = {k: v.clone() for k, v in new._data.items()}
        return new


class StateCache(StateMixin, Cache):
    """
    A cache specifically for BaseState and those that inherit from it,
    mostly to easily call replica to each element inside it
    """

    def __init__(self, value_type: BaseState, **kwargs):
        super().__init__(
            value_type=value_type,   
            **kwargs
        )

    @classmethod
    def of(cls, data:dict):
        return super().of(data, value_type=BaseState)

    def replica(self, detach=False, freeze=False) -> "StateCache":
        """
        Creates a replica of itself using clone, optionally with detach and freeze kwargs.
        Because BaseState will probably have a specific cloning method
        """
        cloned_cache = copy.deepcopy(self)
        cloned_cache._data = {k: v.clone(detach, freeze) for k, v in cloned_cache._data.items()}
        return cloned_cache

    def get_params_flattened(self) -> list[nn.Parameter]:
        """
        Retrieves all params under all keys for user, in list form (so you can pass to optimizer)
        """
        all_params = []
        for s in self.values():
            all_params.extend([param for _, param in s.named_parameters()])
        assert all([isinstance(p, nn.Parameter) for p in all_params])
        return all_params


class StateTimestepCache(StateCache):
    """keys are int type plus more functionalities ig"""

    def __init__(self, value_type: BaseState):
        super().__init__(
            value_type=value_type,
            key_type=int,    
        )

    def values_at(self, timesteps:list[int]) -> "StateTimestepCache":
        new_data = {t:v for t,v in self._data.items() if t in timesteps}
        return StateTimestepCache.of(new_data)
    
    def exclude_timesteps(self, not_timesteps:list[int]) -> "StateTimestepCache":
        """helper method for less verboseness"""
        assert set(not_timesteps).issubset(set(self.keys())), 'StateTimestepCache: Assertion failed. Excluded timesteps must be within' \
            'existing cache timesteps'
        return self.values_at(list(set(self.keys()) - set(not_timesteps)))
