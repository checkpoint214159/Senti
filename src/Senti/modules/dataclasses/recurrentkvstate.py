import copy

import torch
from normal import BaseNormal
from torch import nn


class RecurrentKVMemory(nn.Module):
    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        super().__init__()
        self.key = nn.Parameter(key)
        self.value = nn.Parameter(value)

    def forward(self):
        return self.key, self.value

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
        self.mean_module = RecurrentKVMemory(h_keys, h_values)
        if h_keys_log_std is None:
            h_keys_log_std = torch.ones_like(h_keys) * init_log_std

        if h_values_log_std is None:
            h_values_log_std = torch.ones_like(h_values) * init_log_std
        self.log_std_module = RecurrentKVMemory(h_keys_log_std, h_values_log_std)
    
    @property
    def mean(self) -> torch.Tensor:
        """
        return the mean tensor.
        just take .key .value from the RecurrentKVMemory, stack it along new 0 dim, and yippee
        """
        return torch.stack([self.mean_module.key, self.mean_module.value])

    @property
    def log_std(self) -> torch.Tensor:
        """return the log_std tensor."""
        return torch.stack([self.log_std_module.key, self.log_std_module.value])




if __name__ == "__main__":
    # simple test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RecurrentKVState(device, h_keys=torch.Tensor([1]), h_values=torch.Tensor([1]))
    

