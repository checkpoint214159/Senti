import torch
from torch import nn
from Senti.modules.dataclass.normal import Normal
from omegaconf.dictconfig import DictConfig

class PolicyPredictor(nn.Module):
    """
    Simple class for now to encapsulate a policy predictor that takes
    in a genome and spits out a policy.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.pref_gene_dim = config.pref_gene_dim
        self.policy_dim = config.policy_dim

        self.policy_predictor = nn.Sequential(
            nn.Linear(self.pref_gene_dim, 2 * self.policy_dim),
            nn.LayerNorm(2 * self.policy_dim),
            nn.ReLU(),
            nn.Linear(2 * self.policy_dim, 2 * self.policy_dim)
        )

    def forward(self, x):
        policy = self.policy_predictor(x)
        mean, std = torch.chunk(policy, 2, dim=-1)

        return Normal(
            z_mean=mean,
            z_log_std=std,
            as_parameter=True,
        )