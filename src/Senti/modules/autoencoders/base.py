from abc import ABC, abstractmethod

from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.registry import AUTOENCODERS


@AUTOENCODERS.register_module()
class BaseAutoEncoder(ABC, nn.Module):
    """
    ABC defining methods of autoencoder.
    This base class is meant to house methods that call the actual encoder and decoder,
    so this way we can further modularize each thing seperately and users can init those on their own
    instead if they so choose.

    Everything is left to definition, since a user should define their own autoencoder
    to pluck specific things from the config
    """

    @staticmethod
    @abstractmethod
    def init_encoder(encoder_config: DictConfig | None) -> nn.Module | None:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def init_decoder(decoder_config: DictConfig | None) -> nn.Module | None:
        raise NotImplementedError()

    def encode(self, data):
        return self.encoder(data)

    def decode(self, data):
        return self.decoder(data)
