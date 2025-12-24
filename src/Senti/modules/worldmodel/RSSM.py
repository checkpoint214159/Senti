import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclasses.categorical import GroupedCategoricalState
from Senti.modules.dataclasses.normal import Normal
from Senti.modules.dataclasses.pomdpstate import POMDPState
from Senti.registry import WORLDMODELS


@WORLDMODELS.register_module()
class RSSM(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.z_dim = config.z_dim
        self.h_dim: list[int] = config.h_dim
        self.flattened_h = config.h_dim[0] * config.h_dim[1]
        self.a_dim = config.a_dim
        self.obs_dim = config.obs_dim
        self.hidden = config.hidden_dim
        self.device = config.device
        self.num_GRU_layers = config.num_GRU_layers
        
        self.recurrent_core = nn.GRU(self.z_dim + self.a_dim,
            self.flattened_h,
            num_layers=self.num_GRU_layers,
        )

        self.z_given_h = nn.Sequential(
            nn.Linear(self.flattened_h, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 2 * self.z_dim) # 2x for mean and std
        )

        self.z_given_h_o = nn.Sequential(
            nn.Linear(self.flattened_h + self.obs_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 2 * self.z_dim)
        )

    def initial_h(self, *dims) -> GroupedCategoricalState:
        """
        todo: enable it to be learned? i dont see the point though
        """
        G, C = self.h_dim[-2], self.h_dim[-1]
        dims = dims + (G, C)
        print('dims in initial_h?', dims)
        h_t = torch.zeros(dims, device=self.device)
        return GroupedCategoricalState(h_t)


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

    def forward_h(self, state:POMDPState) -> GroupedCategoricalState:
        """
        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        """
        prev_z, prev_a, prev_h = state.get('z'), state.get('a'), state.get('h')
        z, a = prev_z.sample(), prev_a.sample()
        x = torch.cat([z, a], dim=-1)
        output, h_t = self.recurrent_core(x, prev_h.as_tensor())
        print('h_t out?', h_t.shape)
        G, C = self.h_dim[-2], self.h_dim[-1]
        return GroupedCategoricalState.from_flat_logits(h_t, G, C).to(self.device)

    def forward_z(self, h_t: GroupedCategoricalState, o_embed_t:torch.Tensor | None = None) -> Normal:
        """
        (z_t | h_t) or (z_t | h_t, o_t)
        Still accept the whole POMDPState as an argument, to keep abstraction neat
        """
        is_posterior = False
        print('h_t.as_tensor() shape', h_t.as_tensor().shape)
        if o_embed_t is not None:
            print('o_embed_t shape', o_embed_t.shape)
            is_posterior = True
            x = torch.cat([h_t.as_tensor(), o_embed_t], dim=-1)
            z = self.z_given_h_o(x)
        else:
            z = self.z_given_h(h_t.as_tensor())
        return self.wrap_z(z, is_posterior)
    

