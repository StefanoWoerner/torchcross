import logging
from typing import Callable, Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torchmetrics import MetricCollection

from torchcross import models
from torchcross.cd.metrics import Accuracy, AUROC
from torchcross.cd.metrics.wrappers import CrossDomainMeanWrapper
from torchcross.data.task import TaskDescription
from torchcross.models.lightning.cross_domain import CrossDomainLightningModule


class SimpleCrossDomainClassifier(
    models.SimpleCrossDomainClassifier, CrossDomainLightningModule
):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        optimizer: type[Optimizer],
        lr_scheduler: Callable[[Optimizer], Any] | None = None,
        task_descriptions: list[TaskDescription] = None,
        add_heads_during_training: bool = True,
        scale_loss_by_batch_count: bool = False,
    ):
        super().__init__(
            backbone,
            num_backbone_features,
            task_descriptions,
            add_heads_during_training,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.scale_loss_by_batch_count = scale_loss_by_batch_count

        self.training_metrics = MetricCollection(
            {
                "accuracy": CrossDomainMeanWrapper(Accuracy, task_descriptions),
                "AUROC": CrossDomainMeanWrapper(AUROC, task_descriptions),
            },
            postfix="/train",
        )
        self.validation_metrics = self.training_metrics.clone(postfix="/val")
        self.test_metrics = self.training_metrics.clone(postfix="/test")

    def add_head(self, task_description: TaskDescription) -> nn.Module:
        new_head = super().add_head(task_description)
        self.optimizers().add_param_group({"params": new_head.parameters()})
        return new_head

    def call_metrics(
        self,
        mode: Literal["train", "val", "test"],
        task_description: TaskDescription,
        pred: torch.Tensor,
        y: torch.Tensor,
    ):
        if mode == "train":
            metrics = self.training_metrics
        elif mode == "val":
            metrics = self.validation_metrics
        elif mode == "test":
            metrics = self.test_metrics
        else:
            raise ValueError(f"Invalid mode {mode}")
        return metrics(task_description, pred, y)

    def validation_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        try:
            super().validation_step(batch, batch_idx)
        except ValueError as e:
            if "Head for task" in str(e):
                logging.warning(f"{e} Skipping validation step.")
            else:
                raise e
