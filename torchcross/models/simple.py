from collections import OrderedDict

import torch
from torch import nn

from torchcross.cd.heads import new_classification_head
from torchcross.data.task import TaskDescription
from torchcross.utils.layers import Expand


class SimpleClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        task_description: TaskDescription,
        expand_input_channels: bool = True,
    ) -> None:
        super(SimpleClassifier, self).__init__()

        self.backbone = backbone
        self.num_backbone_features = num_backbone_features
        self.head = new_classification_head(
            task_description.task_target,
            task_description.classes,
            num_backbone_features,
            init="kaiming",
        )
        net_dict = OrderedDict(backbone=self.backbone, head=self.head)
        if expand_input_channels:
            net_dict = OrderedDict(input_module=Expand(-1, 3, -1, -1), **net_dict)
        self.net = torch.nn.Sequential(net_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
