from collections.abc import Callable, Sequence, Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .episodic import EpisodicLightningModule
from ... import models
from ...cd.activations import get_prob_func
from ...cd.losses import get_loss_func
from ...cd.metrics import get_accuracy_func, get_auroc_func
from ...data import Task


class ProtoNet(models.ProtoNet, EpisodicLightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        outer_optimizer: Callable[
            [Iterable[Tensor] | Iterable[dict[str, Any]]], torch.optim.Optimizer
        ],
        distance: Literal[
            "euclidean", "cosine", "squared_euclidean"
        ] = "squared_euclidean",
        outer_lr_scheduler: Callable[[Optimizer], Any] | None = None,
    ) -> None:
        super(ProtoNet, self).__init__(
            backbone,
            num_backbone_features,
            distance=distance,
        )

        self.outer_optimizer = outer_optimizer
        self.outer_lr_scheduler = outer_lr_scheduler
        metric_keys = [
            "accuracy/meta{}/query",
            "AUROC/meta{}/query",
        ]
        self.configure_metrics(metric_keys)

    def compute_metrics(self, task, query_pred):
        accuracy_func = get_accuracy_func(
            task.description.task_target, task.description.classes, self.device
        )
        auroc_func = get_auroc_func(
            task.description.task_target, task.description.classes, self.device
        )
        query_y = task.query[1]
        query_accuracy = accuracy_func(query_pred, query_y)
        query_auroc = auroc_func(query_pred, query_y)
        return query_accuracy, query_auroc

    def get_metrics_and_losses(
        self, batch: Sequence[Task]
    ) -> tuple[list[tuple[Tensor, ...]], list[Tensor], list[Tensor]]:
        metric_values = []
        query_losses = []
        for task in batch:
            support_x, support_y = task.support
            query_x, query_y = task.query

            loss_func = get_loss_func(
                task.description.task_target, task.description.classes, self.device
            )
            pred_func = get_prob_func(task.description.task_target)

            query_logit = self.episode(task)
            query_losses.append(loss_func(query_logit, query_y))
            with torch.no_grad():
                query_pred = pred_func(query_logit)
                metric_values.append(self.compute_metrics(task, query_pred))

        return metric_values, [torch.tensor(0.0) for _ in query_losses], query_losses


class SimpleCrossDomainProtoNet(ProtoNet, models.SimpleCrossDomainProtoNet):
    pass
