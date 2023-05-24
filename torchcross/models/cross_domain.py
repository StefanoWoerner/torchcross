from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

from torchcross.cd.heads import new_classification_head
from torchcross.data.task import TaskDescription
from torchcross.utils.layers import Expand


class CrossDomainModel(nn.Module):
    def __init__(self) -> None:
        super(CrossDomainModel, self).__init__()
        self.register_buffer("device_dummy", torch.empty(0))

    def forward(
        self, x: torch.Tensor, task_description: TaskDescription
    ) -> torch.Tensor:
        net_for_task = self.get_net(task_description)
        return net_for_task(x)

    def get_net(self, task_description: TaskDescription) -> torch.nn.Module:
        raise NotImplementedError


class SimpleCrossDomainClassifier(CrossDomainModel):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        task_descriptions: list[TaskDescription] = None,
        add_heads_during_training: bool = True,
    ) -> None:
        super(SimpleCrossDomainClassifier, self).__init__()

        self.backbone = backbone
        self.num_backbone_features = num_backbone_features

        self.heads = torch.nn.ModuleDict()
        if task_descriptions is None:
            task_descriptions = []
        for task_description in task_descriptions:
            self.add_head(task_description)

        self.add_heads_during_training = add_heads_during_training

    def get_net(self, task_description: TaskDescription) -> torch.nn.Module:
        if task_description.task_identifier not in self.heads:
            if self.training and self.add_heads_during_training:
                self.add_head(task_description)
            else:
                raise ValueError(
                    f"Head for task {task_description.task_identifier} does not exist."
                )
        return torch.nn.Sequential(
            OrderedDict(
                input_module=Expand(-1, 3, -1, -1),
                backbone=self.backbone,
                head=self.heads[task_description.task_identifier],
            )
        )

    def add_head(self, task_description: TaskDescription) -> nn.Module:
        task_identifier = task_description.task_identifier
        if task_identifier in self.heads:
            raise ValueError(f"Head for task {task_identifier} already exists.")
        self.heads[task_identifier] = new_classification_head(
            task_description.task_target,
            task_description.classes,
            self.num_backbone_features,
            self.device_dummy.device,
        )
        return self.heads[task_identifier]
