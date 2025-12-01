import torch
from torch.distributions import Distribution, kl_divergence


class EnergyAggregator:
    """
    computes KL divergence and log-likelihood using torch.distributions.
    Supports:
    - Single distributions
    - Dicts of distributions (e.g., from POMDPState)
    """

    # -----------------------------
    # KL( q || p )
    # -----------------------------
    def kl(
        self,
        q: Distribution | dict[str, Distribution],
        p: Distribution | dict[str, Distribution]
    ):
        """
        q, p: either torch.Distribution or dict[str, Distribution]
        """
        if isinstance(q, Distribution):
            return kl_divergence(q, p).sum()

        total = 0.0
        for key in q.keys():
            total += kl_divergence(q[key], p[key]).sum()
        return total

    # -----------------------------
    # log likelihood: log p(x)
    # -----------------------------
    def log_likelihood(self, x_true, x_pred):
        """
        x_dist: torch.Distribution or dict[str, Distribution]
        observation: tensor or dict[str, tensor]
        """
        if isinstance(x_true, Distribution):
            return x_true.log_prob(x_pred).sum()

        total = 0.0
        for key in x_true.keys():
            total += x_true[key].log_prob(x_pred[key]).sum()
        return total
