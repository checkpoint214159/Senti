import copy
from typing import Iterable

import torch
from torch import nn

from Senti.modules.dataclasses.normal import BaseNormal, Normal


class RecurrentKVState(BaseNormal):
    """
    - recurrent memory state (h_keys, h_values)
        Where the keys and values are retrieved for use in transformer attention
        Both of h_keys and h_values are treated as learned parameters. 
        This troublesome data structure is the entire reason why I had to devise a dataclass,
        because it was too messy to deal with

        both can be of any shape, since we dont really manipulate anything inside of here that is
        shape-dependent, although the convention that I did code this with (if relevant) is 
        Batch, Timestep, Embedding (BTE)

        The base initial state init function from ResidualRecurrentBlock inside wm_lib/util.py
        however we do not offer that here (since it requires information we dont have)
        So please pass in the actual "mean" you want to represent, and we can init the std for you
    """

    def __init__(
        self,
        h_keys: torch.Tensor,
        h_values: torch.Tensor,
        h_keys_log_std: torch.Tensor | None = None,
        h_values_log_std: torch.Tensor | None = None,
        init_log_std: int | None = 1,
    ): 
        super().__init__()

        assert h_keys.shape == h_values.shape
        
        if h_keys_log_std is None:
            h_keys_log_std = torch.ones_like(h_keys) * init_log_std
        self.keys = Normal(
            z_mean=h_keys,
            z_log_std=h_keys_log_std,
        )

        if h_values_log_std is None:
            h_values_log_std = torch.ones_like(h_values) * init_log_std
        self.values = Normal(
            z_mean=h_values,
            z_log_std=h_values_log_std,
        )
    
    @property
    def mean(self) -> torch.Tensor:
        """
        return the mean tensor.
        just take .key .value from the RecurrentKVMemory, stack it along new 0 dim, and yippee
        """
        return torch.stack([self.keys.mean, self.values.mean])

    @property
    def log_std(self) -> torch.Tensor:
        """return the log_std tensor."""
        return torch.stack([self.keys.log_std, self.values.log_std])
    
    @property
    def shape(self):
        return self.keys.shape
    
    @property
    def dim(self):
        return self.keys.dim
    
    def repeat(self, shape:Iterable[int]) -> "RecurrentKVState":
        repeated_k = self.keys.repeat(shape)
        repeated_v = self.values.repeat(shape)
        return RecurrentKVState(
            h_keys=repeated_k.mean,
            h_values=repeated_v.mean,
            h_keys_log_std=repeated_k.log_std,
            h_values_log_std=repeated_v.log_std,
        )  # let torch return errors here

    def concat(self, other, dim:int) -> "RecurrentKVState":
        self.assert_type(other)
        
        return RecurrentKVState(
            h_keys=torch.cat((self.mean_module.key, other.mean_module.key), dim),
            h_values=torch.cat((self.mean_module.value, other.mean_module.value), dim),
            h_keys_log_std=torch.cat((self.log_std_module.key, other.log_std_module.key), dim),
            h_values_log_std=torch.cat((self.log_std_module.value, other.log_std_module.value), dim),
        )  # let torch return errors here

    def compute_energy(self, other: "RecurrentKVState", loss_func):
        self.assert_type(other)
        kE = self.keys.compute_energy(other.keys, loss_func)
        vE = self.values.compute_energy(other.values, loss_func)
        return kE + vE
    
    def raw(self):
        """
        for now, only returns mean of k and v. we aren't sure if we want to train on log_std.
        """
        return self.keys.mean, self.values.mean


if __name__ == "__main__":
    # simple test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RecurrentKVState(device, h_keys=torch.Tensor([1]), h_values=torch.Tensor([1]))
    

