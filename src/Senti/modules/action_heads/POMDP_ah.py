from abc import ABC, abstractmethod

import torch
from omegaconf.dictconfig import DictConfig
from torch import nn

from Senti.modules.action_heads.external import BaseActionHead, ExternalActionHead
from Senti.modules.dataclass.pomdpstate import POMDPState
from Senti.registry import ACTION_HEADS


@ACTION_HEADS.register_module()
class ActionHead(BaseActionHead):
    """
    Basic action head that takes in the current state x and outputs the action taken.
    """
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.config = config
        self.h_dim: list[int] = config.h_dim
        self.flattened_h = config.h_dim[0] * config.h_dim[1]
        self.a_dim = self.config.a_dim
        self.z_dim = config.z_dim
        self.state_dim = self.flattened_h + self.z_dim

        self.output = nn.Sequential(
            nn.Linear(self.state_dim, self.a_dim),
            nn.LayerNorm(self.a_dim),
            nn.ReLU(),
            nn.Linear(self.a_dim, self.a_dim)
        )

    def forward(self, x):
        return self.output(x)
    

    def forward_hz(self, h, z):
        x = torch.cat([h.as_tensor(), z.sample()], dim=-1)
        return self.forward(x)



@ACTION_HEADS.register_module()
class FiLMActionHead(ExternalActionHead):
    """
    Very niche implementation of an action head, with a functional form:
    action = head(x, genome, policy)
    where genome is from our Genome class
    and policy is some externally sampled object

    Forward takes in 3 arguments, and runs them in a specific way. The idea behind FiLM is to
    "apply feature-wise affine transformations to integrate external context into neural network
    activations"
    jargon aside its just h = gamma(gene,pi) * linear(z) + beta(gene, pi)
    where gamma is some scale and beta is some shift
    also we enforce that gene and pi be the same dimensions, so we can concatenate them
    lets only implement something with more overhead like cross-attention if this doesnt really work well
    we got baseclasses anyways
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.config = config
        self.h_dim: list[int] = config.h_dim
        self.flattened_h = config.h_dim[0] * config.h_dim[1]
        self.z_dim = config.z_dim
        self.state_dim = self.flattened_h + self.z_dim
        self.a_dim = self.config.a_dim
        self.pref_gene_dim = self.config.pref_gene_dim
        self.policy_dim = self.config.policy_dim

        self.concat_dim = self.pref_gene_dim + self.policy_dim
        self.modulator = nn.Sequential(
            nn.Linear(self.concat_dim, 2 * self.state_dim),
            nn.LayerNorm(2 * self.state_dim),
            nn.ReLU(),
        )  # scale and shift both are state_dim long, we split after running this module

        self.output = nn.Linear(self.state_dim, self.a_dim)

    def forward_hz(self, h, z, genome, pi):
        x = torch.cat([h.as_tensor(), z.sample()], dim=-1)
        return self.forward(x, genome, pi)

    def forward(self, x, genome, pi):
        modulation = self.modulator(torch.cat([genome, pi], dim=-1))
        gamma, beta = torch.chunk(modulation, 2, dim=-1)
        
        modulated_x = (gamma * x) + beta
        modulated_x = torch.relu(modulated_x)
        
        return self.output(modulated_x)
    
    def forward_state(self, x:POMDPState, genome, pi):
        """
        useful for abstracting logic
        """
        return self.forward(x.get_state(), genome, pi)


