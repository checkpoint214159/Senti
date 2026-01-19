import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclass.normal import Normal
from Senti.registry import PREFERENCES


@PREFERENCES.register_module()
class BasePreferenceHead(nn.Module):
    """
    some kind of preference head to generate a preference over states.
    pretty much just a MLP ig
    simply conditions itself on genome too

    p(z|h, pref_genome)
    """
    def __init__(self, config:DictConfig):
        self.config = config
        self.z_dim = config.z_dim
        self.h_dim = config.h_dim
        self.genome_dim = config.genome_dim
        self.hidden_dim = config.hidden_dim

        self.hg_dim = self.h_dim + self.genome_dim
        self.pref_net = nn.Sequential(
            nn.Linear(self.hg_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 2 * self.z_dim)
        )

    def wrap_z(self, z_mean_logstd, is_posterior=None) -> Normal:
        """wraps z in our normal"""
        mean, log_std = torch.split(z_mean_logstd, self.z_dim, dim=-1)
        log_std = torch.clamp(log_std, min=-5, max=2)
        std = torch.exp(log_std)
        return Normal(
            z_mean=mean,
            z_log_std=std,
            is_posterior=is_posterior,
        )
    
    def forward(self, h, genome):
        x = torch.cat([h, genome], dim=-1)
        z = self.pref_net(x)
        return self.wrap(z, is_posterior=False)
