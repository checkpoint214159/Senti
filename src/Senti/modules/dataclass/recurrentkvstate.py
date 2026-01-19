import copy
from typing import Iterable

import torch
from torch import nn

from Senti.modules.dataclass.normal import BaseNormal, Normal


class RecurrentKVState(BaseNormal):
    """
    - recurrent memory state (h_key, h_value)
        Where the key and value are retrieved for use in transformer attention
        Both of h_key and h_value are treated as learned parameters. 
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
        h_key: torch.Tensor,
        h_value: torch.Tensor,
        h_key_log_std: torch.Tensor | None = None,
        h_value_log_std: torch.Tensor | None = None,
        init_log_std: int | None = 1,
    ): 
        super().__init__()

        assert h_key.shape == h_value.shape
        
        if h_key_log_std is None:
            h_key_log_std = torch.ones_like(h_key) * init_log_std
        self.key = Normal(
            z_mean=h_key,
            z_log_std=h_key_log_std,
        )

        if h_value_log_std is None:
            h_value_log_std = torch.ones_like(h_value) * init_log_std
        self.value = Normal(
            z_mean=h_value,
            z_log_std=h_value_log_std,
        )
    
    @property
    def mean(self) -> torch.Tensor:
        """
        return the mean tensor.
        just take .key .value from the RecurrentKVMemory, stack it along new 0 dim, and yippee
        """
        return torch.stack([self.key.mean, self.value.mean])

    @property
    def log_std(self) -> torch.Tensor:
        """return the log_std tensor."""
        return torch.stack([self.key.log_std, self.value.log_std])
    
    @property
    def shape(self):
        return self.key.shape
    
    @property
    def dim(self):
        return self.key.dim
    
    @classmethod
    def from_normals(self, key, value):
        return RecurrentKVState(
            h_key=key.mean,
            h_value=value.mean,
            h_key_log_std=key.log_std,
            h_value_log_std=value.log_std,
        )  # let torch return errors here
    
    def repeat(self, shape:Iterable[int]) -> "RecurrentKVState":
        return RecurrentKVState.from_normals(self.key.repeat(shape), self.value.repeat(shape))
    
    @classmethod
    def concat(cls, rkvs:list["RecurrentKVState"], dim:int) -> "RecurrentKVState":
        [cls.assert_type(rkv) for rkv in rkvs]
        first_rkv = rkvs[0]
        new_key = type(first_rkv.key).concat([rkv.key for rkv in rkvs], dim)
        new_value = type(first_rkv.value).concat([rkv.value for rkv in rkvs], dim)

        return RecurrentKVState.from_normals(new_key, new_value)

    def compute_energy(self, other: "RecurrentKVState", loss_func):
        self.assert_type(other)
        kE = self.key.compute_energy(other.key, loss_func)
        vE = self.value.compute_energy(other.value, loss_func)
        return kE + vE
    
    def raw(self):
        """
        for now, only returns mean of k and v. we aren't sure if we want to train on log_std.
        """
        return self.key.mean, self.value.mean
    
    def map(self, func):
        return RecurrentKVState.from_normals(self.key.map(func), self.value.map(func))



if __name__ == "__main__":
    # simple test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RecurrentKVState(device, h_key=torch.Tensor([1]), h_value=torch.Tensor([1]))
    

