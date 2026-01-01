from dataclasses import field

import torch

from Senti.modules.dataclasses.base import tensor_node


@tensor_node
class LatentObservation:
    lat_o: torch.Tensor
    allowed_keys: list[str] = field(default_factory=lambda: ['lat_o'])

    def __post_init__(self):
        assert isinstance(self.lat_o, torch.Tensor)
