from collections import OrderedDict
from typing import Literal

import torch
from torch import nn, Tensor

from torchcross.cd.activations import get_log_prob_func
from torchcross.data.task import Task, TaskTarget
from torchcross.models import CrossDomainModel
from torchcross.models.episodic import EpisodicModel
from torchcross.utils.layers import Expand


class ProtoNet(EpisodicModel):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        distance: Literal[
            "euclidean", "cosine", "squared_euclidean"
        ] = "squared_euclidean",
    ) -> None:
        super(ProtoNet, self).__init__()

        self.backbone = backbone
        self.num_backbone_features = num_backbone_features

        self.distance = distance

    def compute_distance(self, x: Tensor, y: Tensor) -> Tensor:
        if self.distance == "euclidean":
            return torch.cdist(x, y)
        elif self.distance == "cosine":
            return 1 - torch.cosine_similarity(x, y)
        elif self.distance == "squared_euclidean":
            return torch.pow(x[:, None] - y[None, :], 2).sum(dim=2)
        else:
            raise ValueError(f"Distance {self.distance} not implemented")

    def episode(self, task: Task) -> torch.Tensor:
        prototypes, neg_prototypes = self.get_prototypes(task)
        query_x = task.query[0]
        query_features = self.backbone(query_x)
        dist = self.compute_distance(query_features, prototypes)
        logits_func = get_log_prob_func(task.description.task_target)
        query_logits = logits_func(-dist)
        if neg_prototypes is not None:
            neg_dist = self.compute_distance(query_features, neg_prototypes)
            neg_logits = logits_func(-neg_dist)
            query_logits = query_logits - neg_logits
        return query_logits

    def get_prototypes(self, task: Task) -> tuple[Tensor, Tensor]:
        support_x, support_y = task.support
        support_features = self.backbone(support_x)
        task_target = task.description.task_target
        classes = task.description.classes

        if task_target is TaskTarget.MULTICLASS_CLASSIFICATION:
            prototypes = torch.stack(
                [support_features[support_y == c].mean(dim=0) for c in classes]
            )
            neg_prototypes = None
        elif task_target is TaskTarget.MULTILABEL_CLASSIFICATION:
            prototypes = torch.stack(
                [support_features[support_y[:, c] == 1].mean(dim=0) for c in classes]
            )
            neg_prototypes = torch.stack(
                [support_features[support_y[:, c] == 0].mean(dim=0) for c in classes]
            )
        elif task_target is TaskTarget.BINARY_CLASSIFICATION:
            prototypes = torch.stack([support_features[support_y == 1].mean(dim=0)])
            neg_prototypes = torch.stack([support_features[support_y == 0].mean(dim=0)])
        elif task_target is TaskTarget.ORDINAL_REGRESSION:
            # treat as multiclass classification for now
            prototypes = torch.stack(
                [support_features[support_y == c].mean(dim=0) for c in classes]
            )
            neg_prototypes = None
        else:
            raise NotImplementedError(f"Task target {task_target} not yet implemented")
        return prototypes, neg_prototypes


class SimpleCrossDomainProtoNet(ProtoNet, CrossDomainModel):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        distance: Literal[
            "euclidean", "cosine", "squared_euclidean"
        ] = "squared_euclidean",
    ) -> None:
        super().__init__(backbone, num_backbone_features, distance)

        self.backbone = torch.nn.Sequential(
            OrderedDict(input_module=Expand(-1, 3, -1, -1), backbone=self.backbone)
        )
