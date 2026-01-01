import copy
from collections import defaultdict
from typing import Any, Type

import torch
from torch import nn

from Senti.modules.dataclasses.base import BaseState


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
            # TODO no need to assert consistency?
            # for v in iterator:
            #     if not isinstance(v, value_type):
            #         raise TypeError(
            #             f"{cls.__name__}.of: inconsistent value types: "
            #             f"{type(v).__name__} vs {value_type.__name__}"
            #         )
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
    
    def set(self, key: Any, value: Any) -> Any:
        self._data[key] = value

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


class TimestepCache(Cache):
    """keys are int type plus more functionalities ig"""

    def __init__(self, value_type: Any):
        super().__init__(
            value_type=value_type,
            key_type=int,    
        )
        self.curr_timestep = 0

    def add(self, key: int, value: Any):
        assert key == self.curr_timestep + 1
        super().add(key, value)
        self.curr_timestep += 1

    def values_at(self, timesteps:list[int]) -> "TimestepCache":
        new_data = {t:v for t,v in self._data.items() if t in timesteps}
        return TimestepCache.of(new_data)
    
    def exclude_timesteps(self, not_timesteps:list[int]) -> "TimestepCache":
        """helper method for less verboseness"""
        assert set(not_timesteps).issubset(set(self.keys())), 'TimestepCache: Assertion failed. Excluded timesteps must be within' \
            'existing cache timesteps'
        return self.values_at(list(set(self.keys()) - set(not_timesteps)))

    def get_latest(self, num:int) -> "TimestepCache":
        """
        helper method to get the latest num timesteps. exists here because we assume sequential timesteps.
        """
        timesteps = list(range(self.curr_timestep - num + 1, self.curr_timestep + 1))
        return self.values_at(timesteps)


class CloneableCache(TimestepCache):
    """
    Narrower typing, only allowing for types that are clonable / have detach types
    """
        
    def clone(self, detach=False):
        """
        Creates a clone, optionally with detach kwarg.
        """
        new = self.__class__(value_type=self.value_type)
        new._data = {k: v.clone() for k, v in self._data.items()}
        if detach:
            new._data = {k: v.detach() for k, v in new._data.items()}
        return new
    

class StateCache(TimestepCache):
    """
    A cache specifically for BaseState and those that inherit from it,
    mostly to easily call replica to each element inside it.

    Note that replica is different from clone, since we cannot call deepcopy for tensors
    as that screws with computational graphs, but we can do so for parameters / modules
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
    

class HierarchicalCache:
    """
    "Higher order cache" to abstract away specific cache instances.
    Note that we make no attempt to 'merge' the caches together: even if all are StateCaches, it is challenging to
    ensure that all of them have the same tiemsteps, and that itself may not be a feature we want.
    
    There are 3 tiers of access to this:
    One is a mapping between a cache name: cache.
    Another is a mapping between key: container, within the cache.
    The final is a mapping between key: semantic, within the container.

    We allow get set has remove for the final two, but only get for the first.
    
    The point of this class is to allow for a 'flat' retrieval, where instead of specifying during
    use 'which cache do I get this semantic from', because at the start we define a unique mapping
    from semantic to cache, we can 'key' by semantic.
    Of course, this runs into the problem of not having duplicate cache types, as the class of the cache
    implicitly states what semantic it can support. So for instance there is no way to distinguish between
    two state caches, as POMDPState only has attribute 's' and not 's_version2.0' in its definition.

    Of course, this can be solved by just having two HierarchicalCaches that represent two different things.
    """
    def __init__(self,
        mapping: dict[str, Cache],
        cache_names: dict[str, Cache] = None,
    ):
        """
        mapping: {'h': self.states, 'z': self.states}
        cache_names: Optional dict to name the caches, e.g. {self.states: "states"}
                       If not provided, we can try to infer names or use generic ones.
                       But it really should be, as it would greatly help with retrieval later.
        """
        self.mapping = mapping
        self.cache_names = cache_names
        
        self.id_cache_names = set({id(cache) for cache in self.cache_names.values()})
        self.unique_caches = set({id(cache) for cache in self.mapping.values()})
        # print('self.unique_caches', self.unique_caches)
        # print('set(self.cache_names.values())', set(self.cache_names.values()))


        assert self.id_cache_names == self.unique_caches, 'Assertion failed. Expect cache aliases equal to unique caches in mapping argument.'
        
        # for cache_obj in self.mapping.values():
        #     # If you provided a name alias, use it. 
        #     # Otherwise, use the class name or a custom 'name' attr if it exists.
        #     unique_attr_name = None
        #     if cache_names and cache_obj in cache_names:
        #         unique_attr_name = cache_names[cache_obj]
        #     else:
        #         unique_attr_name = getattr(cache_obj, "name", f"{type(cache_obj).__name__.lower()}")

        #     assert unique_attr_name in self.unique_caches, f'Assertion failed: got cache name {unique_attr_name} that is not unique.'
            
        #     setattr(self, attr_name, cache_obj)

    def _retrieve_cache_from_semantic(self, semantic: str):
        if semantic not in self.mapping:
            raise KeyError(f"Semantic '{semantic}' is not mapped to any cache.")
    
        return self.mapping[semantic]
    
    def _retrieve_cache(self, cache_name: str):
        if cache_name not in self.cache_names:
            raise KeyError(f"Could not find cache_name '{cache_name}' in {self.cache_names}.")
    
        return self.cache_names[cache_name]

    def get_semantic(self, semantic: str, key: Any) -> Any:
        cache = self._retrieve_cache_from_semantic(semantic)
        if not cache.has(key):
            raise KeyError(f"{key} key does not exist in cache, cannot find container to get")
        container = cache.get(key)
        try:
            return getattr(container, semantic)
        except AttributeError:
            raise AttributeError(
                f"Container {type(container).__name__} at key {key} "
                f"does not have attribute '{semantic}'"
            )

    def set_semantic(self, semantic: str, key: Any, value: Any):
        cache = self._retrieve_cache_from_semantic(semantic)
        if not cache.has(key):
            raise KeyError(f"{key} key does not exist in cache, cannot find container to set")
        
        container = cache.get(key)
        setattr(container, semantic, value)

    def has_semantic(self, semantic: str, key: Any) -> Any:
        cache = self._retrieve_cache_from_semantic(semantic)
        if not cache.has(key):
            return False
        
        container = cache.get(key)
        return hasattr(container, semantic)

    def get_container(self, cache_name: str, key):
        cache = self._retrieve_cache(cache_name)
        return cache.get(key)
    
    def has_container(self, cache_name: str, key):
        cache = self._retrieve_cache(cache_name)
        return cache.has(key)
    
    def set_container(self, cache_name: str, key, value):
        cache = self._retrieve_cache(cache_name)
        return cache.set(key, value)
        
    def remove_container(self, cache_name: str, key):
        cache = self._retrieve_cache(cache_name)
        cache.remove(key)

    def get_params_flattened(self, cache_name: str):
        return self._retrieve_cache(cache_name).get_params_flattened()
    
    def keys(self, cache_name: str):
        return self._retrieve_cache(cache_name).keys()
    
    def get(self, cache_name: str) -> Cache:
        return self._retrieve_cache(cache_name)
    
    def map_cache(self, cache_name: str, func: callable):
        """runs a func on a certain cache and saves the outcome to itself"""

        self.cache_names[cache_name] = func(self.cache_names[cache_name])

    def is_empty(self, cache_name: str):
        return len(self.keys(cache_name)) == 0

    def get_all_at_key(self, key: Any) -> dict[str, Any]:
        results = {}
        for semantic in self.mapping.keys():
            try:
                results[semantic] = self.get(semantic, key)
            except KeyError:
                continue # key might not exist in all caches
        return results
    
    