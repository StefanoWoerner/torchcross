from typing import Callable, Any

import torch
from lightning import pytorch as pl
from torch import Tensor
from torch.optim import Optimizer

from torchcross import models
from torchcross.cd.activations import get_prob_func
from torchcross.cd.losses import get_loss_func
from torchcross.cd.metrics import get_accuracy_func, get_auroc_func
from torchcross.data.task import TaskDescription
from torchcross.models.lightning import mixins


class SimpleClassifier(
    models.SimpleClassifier, mixins.MeanMetricsMixin, pl.LightningModule
):
    def __init__(
        self,
        backbone: tuple[torch.nn.Module, int],
        task_description: TaskDescription,
        optimizer: type[Optimizer],
        lr_scheduler: Callable[[Optimizer], Any] | None = None,
        expand_input_channels: bool = True,
    ) -> None:
        super().__init__(*backbone, task_description, expand_input_channels)

        self.loss_func = get_loss_func(
            task_description.task_target, task_description.classes, self.device
        )
        self.pred_func = get_prob_func(task_description.task_target)
        self.accuracy_func = get_accuracy_func(
            task_description.task_target, task_description.classes, self.device
        )
        self.auroc_func = get_auroc_func(
            task_description.task_target, task_description.classes, self.device
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        metric_keys = ["accuracy/{}", "AUROC/{}"]
        self.configure_metrics(metric_keys)

        self.automatic_optimization = False

    def get_metrics_and_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        pred = self.pred_func(logits)
        return self.compute_metrics(pred, y), loss

    def compute_metrics(self, pred, y):
        accuracy = self.accuracy_func(pred, y)
        auroc = self.auroc_func(pred, y)
        return accuracy, auroc

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        self.optimizers().zero_grad()

        metric_values, loss = self.get_metrics_and_loss(batch)

        self.update_metrics("train", metric_values)

        self.manual_backward(loss)
        self.optimizers().step()

        self.log("loss/train", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.training_metrics)

    def validation_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        metric_values, loss = self.get_metrics_and_loss(batch)

        self.update_metrics("val", metric_values)

        self.log("loss/val", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.validation_metrics, prog_bar=True)

    def test_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        metric_values, loss = self.get_metrics_and_loss(batch)

        self.update_metrics("test", metric_values)

        self.log("loss/test", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
