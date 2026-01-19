import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclass.categorical import GroupedCategoricalState
from Senti.modules.dataclass.normal import Normal
from Senti.registry import DECODERS


@DECODERS.register_module
class LatentObsDecoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.z_dim = config.z_dim
        self.h_dim = config.h_dim
        self.flattened_h = config.h_dim[0] * config.h_dim[1]
        self.obs_dim = config.obs_dim
        self.hidden = config.hidden_dim

        zh_dim = self.z_dim + self.flattened_h
        self.zh_o = nn.Sequential(
            nn.Linear(zh_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.obs_dim) # 2x for mean and std
        )

    def forward(self, z:torch.Tensor, h:torch.Tensor):
        zh = torch.concat([z, h], dim=-1)
        return self.zh_o(zh)
