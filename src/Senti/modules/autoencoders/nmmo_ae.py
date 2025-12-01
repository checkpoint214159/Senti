import torch
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
        super().__init__()
        ae_config = getattr(config, "nmmo_ae")
        encoder_config: DictConfig | None = getattr(ae_config, "encoder", None)
        decoder_config: DictConfig | None = getattr(ae_config, "decoder", None)

        self.encoder: nn.Module | None = self.init_encoder(encoder_config)
        self.decoder: nn.Module | None = self.init_decoder(decoder_config)
        assert encoder_config.hidden_size == decoder_config.hidden_size
        self.hidden_size = encoder_config.hidden_size

    @staticmethod
    def init_encoder(encoder_config: DictConfig | None) -> nn.Module | None:
        name = encoder_config.pop("name")
        device = encoder_config.pop("device")
        return ENCODERS.build(name, **encoder_config).to(torch.device(device))

    @staticmethod
    def init_decoder(decoder_config: DictConfig | None) -> nn.Module | None:
        return None

    def encode(self, data):
        return self.encoder(data)

    def decode(self, data):
        return self.decoder(data)
