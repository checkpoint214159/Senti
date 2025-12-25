import copy
from abc import abstractmethod
from typing import Generic, Iterable, List, Type, TypeVar

import torch
from torch import nn
from torch.distributions import Independent

from Senti.modules.dataclasses.base import BaseState
from Senti.modules.optim.base import BaseLoss


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
    this is a dataclass for normal distributions.
    however for easy of passing the data into various models, we provide
    ways to extract the data into a flattened form, get the mean, get log_std,
    concat with others, etc etc.
    In short the main idea here was to treat this object like a nn.Module, with some
    of the functionality like concat
    
    a crucial piece of data is the posterior boolean we pass in. This allows BaseNormal to
    take on three forms: is_posterior=True, is_posterior=False, is_posterior=None.
    The value of the first two is that we sometimes only conduct operations of certain semantics
    of it being a posterior or prior belief. However if you want to circumvent that entirely
    it can be left as none.
    """
    def __init__(self,
        is_posterior: bool | None = None):
        super().__init__()
        assert isinstance(is_posterior, bool) or is_posterior is None, 'Assertion failed. is_posterior argument passed '\
            'to BaseNormal is neither boolean nor None.'
        self._is_posterior = is_posterior

    @property
    def is_posterior(self):
        if self._is_posterior is None:
            return False
        return self._is_posterior
    
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
        # clamp for numerical stability??
        return torch.exp(torch.clamp(self.log_std, -5, 2))

    @property
    def as_distribution(self):
        return Independent(torch.distributions.Normal(self.mean, self.std), 1) # "1 from the right"

    def sample(self, n=None):
        dist = self.as_distribution
        if n is None:
            return dist.rsample()
        return dist.rsample((n,))

    def log_prob(self, x):
        return self.as_distribution.log_prob(x)  # because of independent, last dimension will be summed across

    @abstractmethod
    def concat(self, other, dim) -> "BaseNormal":
        ...
    
    def as_tensor(self):
        return self.sample()
    
    def entropy(self):
        return self.as_distribution.entropy()


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
        **kwargs
    ):
        super().__init__(**kwargs)
        cls = self.__class__.__name__

        z_init = dim_or_value_init(
            z_mean,
            z_mean_shape,
            init_value=0)
        self.z_mean = nn.Parameter(z_init)

        z_log_std = dim_or_value_init(
            z_log_std,
            z_log_std_shape,
            init_value=log_std_init)
        self.z_log_std = nn.Parameter(z_log_std)

        assert self.z_mean.shape == self.z_log_std.shape, \
            f'{cls}: Assertion failed. z_mean and z_log_std have mismatching shapes: \n' \
            f'z_mean shape: {self.z_mean.shape} \n' \
            f'z_log_std shape: {self.z_log_std.shape}'
        
    @classmethod
    def from_shape(cls, shape, log_std_init=1):
        return cls(z_mean_shape=shape, z_log_std_shape=shape, log_std_init=log_std_init)

    @classmethod
    def from_value(cls, mean, log_std=None, log_std_init=1):
        if log_std is None:
            return cls(z_mean=mean,
                z_log_std_shape=mean.shape, log_std_init=log_std_init)
        return cls(z_mean=mean,
                z_log_std=log_std, log_std_init=log_std_init)

    @property
    def mean(self): return self.z_mean
        
    @property
    def log_std(self): return self.z_log_std
    
    @property
    def shape(self): return self.z_mean.shape
    
    @property
    def dim(self): return self.z_mean.ndim
    
    def compute_energy(self, other, loss_func):
        """
        for computing energy. the other three arguments are placeholder.
        we assume loss_func accepts two distributions as arguments. if they dont,
        loss_func will error out anyways.
        """
        return loss_func(self, other)

    @classmethod
    def concat(cls, normals:list["Normal"], dim:int) -> "Normal":
        assert isinstance(normals, list)
        [cls.assert_type(o) for o in normals]
        return Normal(
            z_mean=torch.concat([n.mean for n in normals], dim),
            z_log_std=torch.concat([n.log_std for n in normals], dim),
        )  # let torch return errors here
    
    @classmethod
    def stack(cls, normals:list["Normal"], dim:int) -> "Normal":
        assert isinstance(normals, list)
        [cls.assert_type(o) for o in normals]
        return Normal(
            z_mean=torch.stack([n.mean for n in normals], dim),
            z_log_std=torch.stack([n.log_std for n in normals], dim),
        )  # let torch return errors here
    
    def repeat(self, shape:Iterable[int]) -> "Normal":
        return Normal(
            z_mean=self.mean.repeat(shape),
            z_log_std=self.log_std.repeat(shape),
        )  # let torch return errors here
    
    def unsqueeze(self, dim:int) -> "Normal":
        return Normal(
            z_mean=self.mean.unsqueeze(dim),
            z_log_std=self.log_std.unsqueeze(dim),
        )  # let torch return errors here
    
    def raw(self):
        """
        its raw type is simply a tuple of mean and log_std. not very useful.
        """
        return self.mean, self.log_std
    
    def map(self, func) -> "Normal":
        return Normal(
            z_mean=func(self.mean),
            z_log_std=func(self.log_std),
        )
    
    @classmethod
    def concat_timesteps(cls, normals):
        "simple helper func to stack along time dim."
        return cls.stack(normals, dim=1)


T = TypeVar("T", bound="BaseNormal")

class DepthNormal(Generic[T], BaseNormal):
    def __init__(
        self,
        depth: int,
        values: List[T],
        stateclass: Type[T],
        **kwargs,
    ):
        """
        the intent for this class, is for each value to have the same shape
        we will assert that each value's shape is equivalent.
        that way we can perform useful operations with that assumption (like getting mean, std, etc)

        depth: extra arg to sanity check
        values: list of values, depth length, of T
        stateclass: T extends BaseState
        of course python isnt strictly typed, but if you are reading this
        you cant blame me for not trying to be clear
        """
        super().__init__(**kwargs)
        cls = self.__class__.__name__
        assert len(values) == depth, f"{cls}: Assertion failed. Passed in list of values not equal to" \
            f"depth. Depth is {depth} len values is {len(values)}"
        assert len(set([v.shape for v in values])) == 1, f"{cls}: Assertion failed. Passed in list of values of unequal shape" \
            f"{[v.shape for v in values]}"
        
        #TODO write assertion to check values are of same shape here!
        self.depth = depth
        self.values = values
        self.stateclass = stateclass

    def clone(self, detach=False, freeze=False) -> "DepthNormal":
        new_node = copy.deepcopy(self)
        new_node.values = [v.clone(detach, freeze) for v in self.values]

        return new_node
    
    @property
    def mean(self):
        """calls for each element in its values, and concats them all."""
        return torch.concat(
            [v.mean for v in self.values],
            dim=0)
    
    @property
    def log_std(self):
        """calls for each element in its values, and concats them all."""
        return torch.concat(
            [v.log_std for v in self.values],
            dim=0)
    
    @property
    def shape(self):
        return self.values[0].shape
    
    @property
    def dim(self):
        return self.values[0].dim
    
    def compute_energy(self, other, loss_func):
        """
        calls loss_func for each. TODO see if we force loss_func to be addable?
        """
        self.assert_type(other)
        assert self.depth == other.depth, 'DepthNormal: Assertion failed. Passed in DepthNormal not equal to depth of self'
        stack = torch.stack(
            [self.values[d].compute_energy(other.values[d], loss_func) for d in range(self.depth)])
        return torch.sum(stack, dim=0)

    @classmethod
    def concat(cls, dns:list["DepthNormal"], dim:int) -> "DepthNormal":
        assert isinstance(dns, list)
        [cls.assert_type(dn) for dn in dns]
        assert len(set([o.depth for o in dns])) == 1, 'DepthNormal: Assertion failed. Passed in list of DepthNormals with unequal depth' \
            'unable to concatenate'
        first_dn = dns[0]
        depth = first_dn.depth
        return DepthNormal(
            depth=depth,
            values=[type(first_dn.values[d]).concat([dn.values[d] for dn in dns], dim) for d in range(depth)],
            stateclass=first_dn.stateclass,
        )
    
    @classmethod
    def stack(cls, dns:list["DepthNormal"], dim:int) -> "DepthNormal":
        assert isinstance(dns, list)
        [cls.assert_type(dn) for dn in dns]
        assert len(set([o.depth for o in dns])) == 1, 'DepthNormal: Assertion failed. Passed in list of DepthNormals with unequal depth' \
            'unable to concatenate'
        first_dn = dns[0]
        depth = first_dn.depth
        return DepthNormal(
            depth=depth,
            values=[type(first_dn.values[d]).stack([dn.values[d] for dn in dns], dim) for d in range(depth)],
            stateclass=first_dn.stateclass,
        )
    
    def repeat(self, shape:Iterable[int]) -> "DepthNormal":
        return DepthNormal(
            depth=self.depth,
            values=[self.values[d].repeat(shape) for d in range(self.depth)],
            stateclass=self.stateclass,
        )
    
    def unsqueeze(self, dim:int) -> "DepthNormal":
        return DepthNormal(
            depth=self.depth,
            values=[self.values[d].unsqueeze(dim) for d in range(self.depth)],
            stateclass=self.stateclass,
        )
    
    def raw(self):
        """
        simply list of raws of values
        """
        return [v.raw() for v in self.values]
    
    def map(self, func) -> "DepthNormal":
        return DepthNormal(
            depth=self.depth,
            values=[self.values[d].map(func) for d in range(self.depth)],
            stateclass=self.stateclass,
        )

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
