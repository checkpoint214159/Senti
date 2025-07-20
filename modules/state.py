import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class StateNode(nn.Module):
    def __init__(
        self,
        state_value: torch.Tensor | None = None,
        state_dim: int | None = None,
        discrete=False,
        init_std=1.0,
        min_std=1e-3,
        learn_std=False,
        device=None
    ):
        """
        Wraps around an existing state tensor.

        Args:
            state_value (torch.Tensor or None): State value to use, if None tries to init with torch zeros
            state_dim (int or None): If state_value is None, fallback to torch zeros init
            discrete (bool): If True, uses discrete state representation.
            init_std (float): Initial std for continuous case.
            min_std (float): Minimum std for numerical stability.
            learn_std (bool): Whether std is a learnable parameter.
            device (str or torch.device): Device to place parameters on.
        """
        assert not (state_value is None and state_dim is None), 'Assertion failed, StateNode recieved a value and a default init dimension, failing gracefully.'
        if state_value is not None:
            assert isinstance(state_value, torch.Tensor), 'Assertion failed, expected state_value to be a Tensor, failing gracefully.'
        super().__init__()
        self.not_default = True if state_value is not None else False
        self.discrete = discrete
        self.min_std = min_std
        self.device = state_value
        self.logits, self.mean = None, None

        if self.discrete:
            self.logits = nn.Parameter(state_value) if self.not_default \
                else nn.Parameter(torch.zeros(state_dim))
        else:
            self.mean = nn.Parameter(state_value) if self.not_default \
                else nn.Parameter(torch.zeros(state_dim))
            if learn_std:
                self._log_std = nn.Parameter(torch.ones(state_dim, device=device) * torch.log(torch.tensor(init_std)))
            else:
                self.register_buffer("_log_std", torch.ones(state_dim) * torch.log(torch.tensor(init_std)))
    
    def get_distribution(self):
        """Return a torch.distributions object for sampling or computing log-probs."""
        if self.discrete:
            return Categorical(logits=self.logits)
        else:
            std = torch.clamp(self._log_std.exp(), min=self.min_std)
            return Normal(loc=self.mean, scale=std)

    def sample(self):
        """Sample from the state distribution."""
        dist = self.get_distribution()
        return dist.sample()

    def mode(self):
        """Return the mode of the distribution (argmax or mean)."""
        dist = self.get_distribution()
        return dist.probs.argmax(dim=-1) if self.discrete else dist.mean

    def entropy(self):
        """Compute the entropy of the distribution."""
        return self.get_distribution().entropy()

    def log_prob(self, value):
        """Compute log-probability of a given sample."""
        return self.get_distribution().log_prob(value)

    def forward(self):
        """Return current parameter (logits or mean/std)."""
        if self.discrete:
            return self.logits
        else:
            std = torch.clamp(self._log_std.exp(), min=self.min_std)
            return self.mean, std
        
    def __repr__(self):
        if self.discrete:
            probs = F.softmax(self.logits, dim=-1)
            topk = torch.topk(probs, k=min(3, probs.numel()))
            return (
                f"StateNode(discrete=True, dim={self.logits.shape[0]}, "
                f"top_probs={topk.values.tolist()}, indices={topk.indices.tolist()})"
            )
        else:
            mean = self.mean.detach().cpu().numpy()
            std = torch.clamp(self._log_std.exp(), min=self.min_std).detach().cpu().numpy()
            return (
                f"StateNode(discrete=False, dim={self.mean.shape[0]}, "
                f"mean={mean.tolist()}, std={std.tolist()})"
            )