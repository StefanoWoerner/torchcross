import math
from collections import OrderedDict
from collections.abc import Sequence, Callable
from typing import Optional, Literal, Tuple, List

import torch
import torchopt
from torch import nn, Tensor

from torchcross.cd.heads import new_classification_head
from torchcross.cd.losses import get_loss_func
from torchcross.data.metadataset.few_shot import get_indices
from torchcross.data.task import TaskDescription, Task, TaskTarget
from torchcross.models.episodic import EpisodicModel


class ProtoNet(EpisodicModel):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        distance: Literal["euclidean", "cosine", "squared_euclidean"] = "euclidean",
    ) -> None:
        super(ProtoNet, self).__init__()

        self.backbone = backbone
        self.num_backbone_features = num_backbone_features

        self.distance = distance

    def episode(self, task: Task) -> torch.Tensor:
        prototypes, neg_prototypes = self.get_prototypes(task)
        query_x = task.query[0]
        query_features = self.backbone(query_x)
        query_logits = torch.cdist(query_features, prototypes)

        return query_logits

    def get_prototypes(self, task: Task) -> tuple[Tensor, Tensor]:
        support_x, support_y = task.support
        support_features = self.backbone(support_x)
        task_target = task.task_target
        classes = task.classes

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
            raise NotImplementedError("Ordinal regression not yet implemented")
        else:
            raise NotImplementedError(f"Task target {task_target} not yet implemented")
        return prototypes, neg_prototypes
