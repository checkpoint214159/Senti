from abc import ABC, abstractmethod

import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.registry import AUTOENCODERS, DECODERS, ENCODERS


@AUTOENCODERS.register_module()
class BaseAutoEncoder(ABC, nn.Module):
    """
    ABC defining methods of autoencoder.
    This base class is meant to house methods that call the actual encoder and decoder,
    so this way we can further modularize each thing seperately and users can init those on their own
    instead if they so choose.

    provided some default utilities of using name and device, but this isnt strict
    """

    @staticmethod
    def init_encoder(encoder_config: DictConfig | None) -> nn.Module | None:
        name = encoder_config.pop("name")
        device = encoder_config.pop("device")
        return ENCODERS.build(name, **encoder_config).to(torch.device(device))

    @staticmethod
    def init_decoder(decoder_config: DictConfig | None) -> nn.Module | None:
        name = decoder_config.pop("name")
        device = decoder_config.pop("device")
        return DECODERS.build(name, **decoder_config).to(torch.device(device))

    def encode(self, data):
        raise NotImplementedError()

    def decode(self, data):
        raise NotImplementedError()
