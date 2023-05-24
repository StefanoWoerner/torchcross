from typing import Sequence

import torch
from torch import nn

from torchcross.data import Task


class EpisodicModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Sequence[Task]) -> list[torch.Tensor]:
        return [self.episode(task) for task in batch]

    def episode(self, task: Task) -> torch.Tensor:
        ...
