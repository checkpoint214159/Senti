import torch
import torch.nn as nn
from torch.distributions import Independent, OneHotCategorical
from einops import rearrange
from Senti.modules.dataclass.base import BaseState


class GroupedCategoricalState(BaseState):
    """
    Presumes a tensor shape of [..., G, C]. When as_tensor is called, we will merge the final two to get 
    [..., GC] shaped tensor (alongside other semantics like sampling).
    """

    def __init__(self,
        logits: torch.Tensor,
        as_parameter: bool = True,
        **kwargs
    ):
        super().__init__(as_parameter, **kwargs)
        assert len(logits.shape) >= 2, "Assertion failed. logits must have 2 dimensions minimum, where last two "\
            "are assumed to be num_groups, num_classes"
        self._kwargs = kwargs
        self.logits = nn.Parameter(logits) if as_parameter else logits
        self.num_groups = logits.shape[-2]
        self.num_classes = logits.shape[-1]

    @property
    def shape(self):
        return f"GroupedCategoricalState of overall shape {self.logits.shape}, {self.num_groups} groups, {self.num_classes} categories"

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
        optionally pass in argument to sample.
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
    
    def __flatten__(self):
        children = (self.logits, )
        aux_data = (None, )
        return children, aux_data
    
    @classmethod
    def __unflatten__(cls, children, aux_data):
        return cls(
            logits=children[0], 
            as_parameter=False 
        )
    
    def parameterize(self) -> "GroupedCategoricalState":
        return self.__class__(
            logits=self.logits,
            as_parameter=True,
            **self._kwargs # Pass the original flags back in
        )

class DepthGroupedCategoricalState(GroupedCategoricalState):
    """
    Inherit from GroupedCategoricalState, but asserts for the particular shape
    [Depth, ..., G, C]. as_tensor makes this [..., (Depth * GC)] as a result.
    """
    def __init__(self,
        logits: torch.Tensor,
        as_parameter: bool = True,
        **kwargs,
    ):
        super(GroupedCategoricalState, self).__init__(as_parameter)
        assert len(logits.shape) >= 3, "Assertion failed. logits must have 3 dimensions minimum, where last two "\
            "are assumed to be num_groups, num_classes, and the first dimension is the depth of the state."
        self.logits = nn.Parameter(logits) if as_parameter else logits
        self.num_groups = logits.shape[-2]
        self.num_classes = logits.shape[-1]
        self.depth = logits.shape[0]

        self._kwargs = kwargs

    def as_tensor(self, use_sample=False, flatten_depth=True):
        out = super().as_tensor(use_sample)
        
        if flatten_depth:
            return rearrange(out, "L ... GC -> ... (L GC)")
        else:
            return out