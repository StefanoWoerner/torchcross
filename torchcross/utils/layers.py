import torch
import torch.nn as nn


class Expand(nn.Module):
    """Uses `torch.Tensor.expand` to expand a tensor to a given size.

    Args:
        sizes: The sizes to expand the tensor to.
    """

    def __init__(self, *sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(*self.sizes)
