from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.registry import AUTOENCODERS, ENCODERS

from .base import BaseAutoEncoder


@AUTOENCODERS.register_module()
class NmmoAE(BaseAutoEncoder):
    """
    Specifically fishes out nmmo_encoder and nmmo_decoder from config.
    """
    def __init__(self, config: DictConfig):
        encoder_config: DictConfig | None = getattr(config, "nmmo_ae_encoder", None)
        decoder_config: DictConfig | None = getattr(config, "nmmo_ae_decoder", None)
        if encoder_config is None and decoder_config is None:
            raise AttributeError("Config does not have encoder or decoder field. Must at least have one.")

        self.encoder: nn.Module | None = self.init_encoder(encoder_config)
        self.decoder: nn.Module | None = self.init_decoder()

    @staticmethod
    def init_encoder(encoder_config: DictConfig | None) -> nn.Module | None:
        return ENCODERS.build(encoder_config.name, encoder_config)

    @staticmethod
    def init_decoder(decoder_config: DictConfig | None) -> nn.Module | None:
        return None

    def encode(self, data):
        return self.encoder(data)

    def decode(self, data):
        return self.decoder(data)
