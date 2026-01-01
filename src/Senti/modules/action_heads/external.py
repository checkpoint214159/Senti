from abc import ABC, abstractmethod

import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.action_heads.base import BaseActionHead
from Senti.registry import ACTION_HEADS


@ACTION_HEADS.register_module()
class ExternalActionHead(BaseActionHead):
    """
    For an action head that has external parameters.
    The reasoning behind this is that these parameters might not be learned 
    or backpropped for, i.e they are handled by something else externally.
    An example would be my planned idea of a 'gene', which is something that is not optimized
    by the agent but instead by an external selection process, hence we do not take gradient on it

    The above is the ideal semantic, but of course the implementation is necessarily vague and because
    of that, we force implementors to implement a forward method with a specific signature
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def forward(self, x: nn.Module, *externals):
        raise NotImplementedError("Forward method of ExternalActionHead not implemented")

    