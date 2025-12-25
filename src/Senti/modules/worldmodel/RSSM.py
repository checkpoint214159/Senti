import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.dataclasses.categorical import GroupedCategoricalState
from Senti.modules.dataclasses.normal import Normal
from Senti.modules.dataclasses.pomdpstate import POMDPState
from Senti.registry import WORLDMODELS


@WORLDMODELS.register_module()
class RSSM(nn.Module):
    def __init__(self,
            config: DictConfig):
        """
        special h_aggregator module (handled externally) that helps us aggregate from
        [B, D, E] into [B, E], basically across depths.
        """
        super().__init__()

        self.z_dim = config.z_dim
        self.h_dim: list[int] = config.h_dim
        self.flattened_h = config.h_dim[0] * config.h_dim[1]
        self.a_dim = config.a_dim
        self.obs_dim = config.obs_dim
        self.hidden = config.hidden_dim
        self.device = config.device
        self.num_GRU_layers = config.num_GRU_layers
        self.h_groups, self.h_classes = self.h_dim[-2], self.h_dim[-1]
        
        self.recurrent_core = nn.GRU(self.z_dim + self.a_dim,
            self.flattened_h,
            num_layers=self.num_GRU_layers,
            batch_first=True
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
        G, C = self.h_groups, self.h_classes
        dims = dims + (G, C)
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
        
        notation:
        B = batch size
        L = sequence length
        D = 2 if bidirectional=True, 1 otherwise. SHould always be 1 TODO lazy to account for this now
        N = num_layers
        """
        G, C = self.h_groups, self.h_classes
        prev_z, prev_a, prev_h = state.get('z'), state.get('a'), state.get('h')
        z, a = prev_z.sample(), prev_a.sample()
        x = torch.cat([z, a], dim=-1)  # [B, L, az_emb]
        prev_h = prev_h.as_tensor()  # [B, T, h_emb]. T SHOULD be 1 here.
        assert prev_h.shape[1] == 1, 'Assertion failed. T should be 1 only, since it is the seed'
        prev_h = prev_h.squeeze(1)  # [B, h_emb]
        prev_h = prev_h.unsqueeze(0).repeat(self.num_GRU_layers, 1, 1)  # adds layer, [N, B, h_emb]
        output, h_t = self.recurrent_core(x, prev_h)  # output: [B, L, D*h_emb], h: [D*N, B, h_emb]

        return GroupedCategoricalState.from_flat_logits(output, G, C).to(self.device), \
            GroupedCategoricalState.from_flat_logits(h_t, G, C).to(self.device)

    def forward_z(self, h_t: GroupedCategoricalState, o_embed_t:torch.Tensor | None = None) -> Normal:
        """
        (z_t | h_t) or (z_t | h_t, o_t)
        Still accept the whole POMDPState as an argument, to keep abstraction neat
        """
        is_posterior = False
        if o_embed_t is not None:
            is_posterior = True
            h_t = h_t.as_tensor()
            print('h_t shape?', h_t.shape)
            print('o_embed_t shape?', o_embed_t.shape)
            x = torch.cat([h_t, o_embed_t], dim=-1)
            z = self.z_given_h_o(x)
        else:
            z = self.z_given_h(h_t.as_tensor())
        return self.wrap_z(z, is_posterior)
    
    
    

