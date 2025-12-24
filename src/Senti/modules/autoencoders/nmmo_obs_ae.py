import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclasses.normal import Normal
from Senti.registry import AUTOENCODERS, ENCODERS

from .base import BaseAutoEncoder


@AUTOENCODERS.register_module()
class NmmoObsAE(BaseAutoEncoder):
    """
    Specifically fishes out nmmo_obs_encoder and nmmo_obs_decoder from config.
    Unifies their functionality, and wraps them to combine with our dataclass
    """
    def __init__(self, ae_config: DictConfig):
        super().__init__()
        encoder_config: DictConfig | None = getattr(ae_config, "encoder", None)
        decoder_config: DictConfig | None = getattr(ae_config, "decoder", None)

        self.encoder: nn.Module | None = self.init_encoder(encoder_config)
        # self.decoder: nn.Module | None = self.init_decoder(decoder_config)
        assert encoder_config.hidden_size == decoder_config.hidden_size
        self.hidden_size = encoder_config.hidden_size

    def encode(self, data):
        agent_embeddings, my_agent_embeddings = self.encoder(data)
        # force as normal for now
        # NOTE: LEARN VARIANCE BOOKMARK HERE
        return agent_embeddings, my_agent_embeddings

    def decode(self, data):
        # TODO figure out the decoder version of this later.
        raise NotImplementedError()
