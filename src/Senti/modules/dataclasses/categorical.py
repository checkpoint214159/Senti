import torch
import torch.nn as nn
from torch.distributions import Independent, OneHotCategorical

from Senti.modules.dataclasses.base import BaseState


class GroupedCategoricalState(BaseState):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        assert len(logits.shape) >= 3, "Assertion failed. logits must have 3 dimensions minimum, where last two "\
            "are assumed to be num_groups, num_classes"
        self.logits = nn.Parameter(logits)
        self.num_groups = logits.shape[-2]
        self.num_classes = logits.shape[-1]

    @classmethod
    def from_flat_logits(cls, flat_logits: torch.Tensor, num_groups: int, num_classes: int):
        """
        factory method, provide groups and classes so we can view it
        """
        batch_dims = flat_logits.shape[:-1] 
        reshaped = flat_logits.view(*batch_dims, num_groups, num_classes)
        return cls(reshaped)

    def as_distribution(self):
        base_dist = OneHotCategorical(logits=self.logits)
        return Independent(base_dist, 1)  # "1 from the right"

    def sample(self):
        """
        straight-through gradient trick
        sampling from a categorical results in collapse to one-hot vectors
        """
        dist = self.as_distribution()
        stoch = dist.sample()
        
        probs = torch.softmax(self.logits, dim=-1)
        return stoch + probs - probs.detach()

    def as_tensor(self, use_sample=False):
        """
        optionally pass in argument to sample
        """
        if use_sample:
            x = self.sample()
        else:
            x = torch.softmax(self.logits, dim=-1)
        return x.flatten(start_dim=-2)

    def raw(self):
        return torch.argmax(self.logits, dim=-1)
    
    def entropy(self):
        return self.as_distribution().entropy()

    # def compute_energy(self, other: "GroupedCategoricalState", loss_func=None):
    #     """
    #     In Active Inference, the energy (VFE) between two categorical 
    #     states is often the KL Divergence.
    #     """
    #     self.assert_type(other)
    #     p = self.as_distribution()
    #     q = other.as_distribution()
    #     return torch.distributions.kl.kl_divergence(p, q)