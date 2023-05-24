import logging
from typing import Callable, Any

import torch
from torch import nn
from torch.optim import Optimizer

from torchcross import models
from torchcross.cd.metrics import get_accuracy_func, get_auroc_func
from torchcross.data.task import TaskDescription
from torchcross.models.lightning.cross_domain import CrossDomainLightningModule


class SimpleCrossDomainClassifier(
    models.SimpleCrossDomainClassifier, CrossDomainLightningModule
):
    def __init__(
        self,
        backbone: tuple[torch.nn.Module, int],
        optimizer: type[Optimizer],
        lr_scheduler: Callable[[Optimizer], Any] | None = None,
        task_descriptions: list[TaskDescription] = None,
        add_heads_during_training: bool = True,
    ):
        super().__init__(
            *backbone,
            task_descriptions,
            add_heads_during_training,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        metric_keys = ["accuracy/{}", "AUROC/{}"]
        self.configure_metrics(metric_keys)

    def add_head(self, task_description: TaskDescription) -> nn.Module:
        new_head = super().add_head(task_description)
        self.optimizers().add_param_group({"params": new_head.parameters()})
        return new_head

    def compute_metrics(self, task_description, pred, y):
        accuracy_func = get_accuracy_func(
            task_description.task_target, task_description.classes, self.device
        )
        auroc_func = get_auroc_func(
            task_description.task_target, task_description.classes, self.device
        )
        accuracy = accuracy_func(pred, y)
        auroc = auroc_func(pred, y)
        return accuracy, auroc

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
