import base64

import numpy as np
import torch


def decode_base64(obj, device=None):
    """
    Decode JSON objects encoded via `encode_base64` back into torch.Tensor.
    """
    if isinstance(obj, dict):
        if {"data", "dtype", "shape"} <= obj.keys():
            arr = np.frombuffer(
                base64.b64decode(obj["data"]), dtype=np.dtype(obj["dtype"])
            ).reshape(obj["shape"])
            return torch.from_numpy(arr).to(device) if device else torch.from_numpy(arr)
        else:
            return {k: decode_base64(v, device=device) for k, v in obj.items()}
    
    if isinstance(obj, list):
        return [decode_base64(v, device=device) for v in obj]
    
    return obj


def decode_torch_dtype(obj):
    if isinstance(obj, dict):
        # dtype descriptor?
        if {"dtype", "shape", "offset", "size"} <= obj.keys():
            return (
                getattr(torch, obj["dtype"]),
                tuple(obj["shape"]),
                obj["offset"],
                obj["size"],
            )
        else:
            return {k: decode_torch_dtype(v) for k, v in obj.items()}
    
    return obj
