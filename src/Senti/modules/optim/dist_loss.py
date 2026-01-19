from typing import Any

import torch
from torch.distributions import Distribution, kl_divergence

from Senti.modules.dataclass.normal import BaseNormal
from Senti.modules.optim.base import BaseLoss


class DistributionLosses(BaseLoss):
    """
    computes KL divergence and log-likelihood using torch.distributions.
    Supports:
    - Single distributions
    - Dicts of distributions (e.g., from POMDPState)
    """
    types = [BaseNormal]

    def __init__(self):
        super().__init__()
        

    @classmethod
    def supports_type(cls, t) -> bool:
        return any(isinstance(t, v) for v in cls.types)

    # -----------------------------
    # KL( q || p )
    # -----------------------------
    def kl(
        self,
        q: BaseNormal,
        p: BaseNormal,
    ):
        """
        q, p: either torch.Distribution or dict[str, Distribution]
        """
        assert self.supports_type(q), 'DistributionLosses: Assertion failed. q is not in valid types.'
        assert self.supports_type(p), 'DistributionLosses: Assertion failed. p is not in valid types.'
        
        return kl_divergence(q.as_distribution, p.as_distribution).sum()


    def log_likelihood(
            self,
            x_true: BaseNormal,
            x_pred: torch.Tensor,
        ):
        """
        x_true: torch.Distribution or dict[str, Distribution]
        x_pred: tensor or dict[str, tensor]
        """
        assert self.supports_type(x_true), \
            'DistributionLosses: Assertion failed. x_true is not in valid types.'

        return x_true.as_distribution.log_prob(x_pred).sum()
    
    def MSE(
        self,
        pred: torch.Tensor,
        true: torch.Tensor,
    ):
        
        return ((true - pred) ** 2).sum()
