import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclass.normal import Normal
from Senti.registry import AUTOENCODERS

from .base import BaseAutoEncoder


@AUTOENCODERS.register_module()
class zAE(BaseAutoEncoder):
    """
    serves to wrap a simple sequential with our dataclasses functionality.
    """
    def __init__(self, ae_config: DictConfig):
        super().__init__()
        
        obs_hid_size = ae_config.obs_hid_size
        z_dim = ae_config.z_dim
        self.z_encoder = nn.Sequential(
            nn.Linear(obs_hid_size, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU()
        )  # P(z|o)
        # maps states to latent observation
        self.z_decoder = nn.Sequential(
            nn.Linear(z_dim, obs_hid_size),
            nn.LayerNorm(obs_hid_size),
            nn.ReLU()
        ) # P(o|z).

    def encode(self, data: Normal):
        # NOTE: LEARN VARIANCE BOOKMARK HERE
        return Normal.from_value(self.z_encoder(data.mean))
    
    def decode(self, data: Normal):
        return Normal.from_value(self.z_decoder(data.mean))


