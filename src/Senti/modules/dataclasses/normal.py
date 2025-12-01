import copy
from abc import abstractmethod

import torch
from base import BaseState
from torch import nn


def dim_or_value_init(
        v: torch.Tensor | None = None,
        v_shape: tuple[int] | None = None,
        init_value: int = 1,
    ) -> torch.Tensor:
    assert (v is not None or v_shape is not None), \
        'Assertion failed. Must at least provide one of v OR v_shape.'
    assert not (v is not None and v_shape is not None), \
        'Assertion failed. Must only provide one z_mean or v_shape, not both!'
    
    if v is not None:
        v_init = v
    else:
        v_init = torch.ones(*v_shape) * init_value
    
    return v_init

class BaseNormal(BaseState):
    """
    implementors to provide:
      - mean (Tensor)
      - log_std or std (Tensor)
    """
    def __init__(self):
        super().__init__()
    
    @property
    @abstractmethod
    def mean(self):
        """return the mean tensor."""
        ...

    @property
    @abstractmethod
    def log_std(self):
        """return the log_std tensor."""
        ...

    @property
    def std(self):
        return torch.exp(self.log_std)

    def as_distribution(self):
        return torch.distributions.Normal(self.mean, self.std)

    def sample(self, n=None):
        dist = self.as_distribution()
        if n is None:
            return dist.rsample()
        return dist.rsample((n,))

    def log_prob(self, x):
        return self.as_distribution().log_prob(x)


class Normal(BaseNormal):
    """
    for a user to provide is either its dimension or its values, then infer from there
    
    this one presumes no custom data types or unique keys (see RecurrentKVState which has specific
    encapsulates private 'keys' and 'values' attributes), and offers the flexibility for users to define
    only either the values (of mean and log_std) or the shape (of mean and log_std).
    """
    def __init__(
        self,
        z_mean: torch.Tensor | None = None,
        z_mean_shape: tuple[int] | None = None,
        z_log_std: torch.Tensor | None = None,
        z_log_std_shape: tuple[int] | None = None,
        log_std_init: int = 1,
    ):
        super().__init__()
        cls = self.__class__.__name__

        z_init = dim_or_value_init(
            z_mean,
            z_mean_shape,
            init_value=0)
        self.z_mean = nn.Parameter(z_init)

        z_log_std_init = dim_or_value_init(
            z_log_std,
            z_log_std_shape,
            init_value=log_std_init)
        self.z_log_std = nn.Parameter(z_log_std_init)

        assert self.z_mean.shape == self.z_log_std.shape, \
            f'{cls}: Assertion failed. z_mean and z_log_std have mismatching shapes: \n' \
            f'z_mean shape: {self.z_mean.shape} \n' \
            f'z_log_std shape: {self.z_log_std.shape}'
        
    @classmethod
    def from_shape(cls, shape, init_log_std=1):
        return cls(z_mean_shape=shape, z_log_std_shape=shape, log_std_init=init_log_std)

    @classmethod
    def from_value(cls, mean, log_std=None):
        return cls(z_mean=mean, z_log_std=log_std)

    @property
    def mean(self):
        return self.z_mean
        
    @property
    def log_std(self):
        return self.z_log_std

if __name__ == "__main__":
    # simple test
    Normal(z_mean=torch.Tensor([1]), z_log_std=torch.Tensor([1]))
    Normal.from_value(mean=torch.Tensor([1]), log_std=torch.Tensor([1]))
    Normal(z_mean_shape=(1,2,3), z_log_std_shape=(1,2,3))
    Normal.from_shape((1,2,3))
    Normal(z_mean=torch.Tensor([1]), z_log_std_shape=(1,))
    tests = [
        lambda: Normal(z_mean=torch.Tensor([1])),
        lambda: Normal(z_mean_shape=(1,2,3)),
        lambda: Normal(z_mean=torch.Tensor([1]), z_log_std=torch.Tensor([1, 2])),
        lambda: Normal(z_mean_shape=(1,2,3), z_log_std_shape=(1,2,4)),
    ]

    for i, test in enumerate(tests):
        try:
            test()
        except Exception as e:
            print(f"Test {i} failed:", e)
