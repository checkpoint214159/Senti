from abc import abstractmethod

import torch
from base import BaseState


class BaseDiscrete(BaseState):
    """
    Base class for discrete-valued states.
    Requires implementors to provide:
      - probs or logits
      - as_distribution()
    """

    @property
    @abstractmethod
    def probs(self):
        """Return probability vector over categories."""
        ...

    def as_distribution(self):
        return torch.distributions.Categorical(probs=self.probs)

    def sample(self, n=None):
        dist = self.as_distribution()
        if n is None:
            return dist.sample()
        return dist.sample((n,))

    def log_prob(self, x):
        return self.as_distribution().log_prob(x)
