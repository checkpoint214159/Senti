from abc import ABC, abstractmethod

import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.registry import ACTION_HEADS


@ACTION_HEADS.register_module()
class BaseActionHead(ABC, nn.Module):
    """
    Base class to encapsulate the idea of an action head.
    Mostly just to enforce certain argument and attribute names so we can keep it standardized
    Assumes that you are operating on tensor of shape [..., n] and want to map it to [..., m]
    basically not caring about the front dimensions.
    You don't need to inherit from this ABC though, since this semantic is pretty flexible

    Also note that action heads are semantically different from an 'action decoder'
    One predicts 'what action to take' in some latent action space, whilst the decoder decodes
    this to the actual physical action space
    You can encapsulate the latter in the former
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.x_dim = config.x_dim
        self.a_dim = config.a_dim
