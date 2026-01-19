import copy
from collections import defaultdict

import torch

from Senti.modules.dataclass.base import BaseState


class TensorMixin:
    """clone/detach behaviour"""
    def clone(self, detach=False):
        self._data = {k: v.clone() for k, v in self._data.items()}
        if detach:
            self._data = {k: v.detach() for k, v in self._data.items()}
        return self


# class StateMixin:
#     """replica behaviour for specifically State like things"""
#     def replica(self, detach=False, freeze=False):
#         cloned_cache = copy.deepcopy(self)
#         cloned_cache._data = {k: v.clone(detach, freeze) for k, v in cloned_cache._data.items()}
#         return cloned_cache


class TimestepMixin:
    """timestep-aware functions"""
    def get_timesteps(self, timesteps: list) -> list[BaseState]:
        return [self.get(t) for t in timesteps]

    # def get_all_h_states(self):
    #     # TODO trash function generalize it to not use get_h_states.
    #     depthwise_all_s = defaultdict(list)
    #     depthwise_stacked_result = {}

    #     for s in self.values():
    #         for depth, s_tuple in enumerate(s.get_h_states()):
    #             depthwise_all_s[depth].append(s_tuple)

    #     for depth, s_tuples in depthwise_all_s.items():
    #         length_of_tuple = len(s_tuples[0])
    #         depthwise_stacked_result[depth] = tuple(
    #             [torch.concat([s_tuple[i] for s_tuple in s_tuples])
    #              for i in range(length_of_tuple)]
    #         )

    #     return depthwise_stacked_result

