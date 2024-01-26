import torch
from lightning import pytorch as pl
from torch import Tensor

from torchcross import models
from torchcross.cd.activations import get_prob_func
from torchcross.cd.losses import get_loss_func
from torchcross.data.task import TaskDescription
from torchcross.models.lightning import mixins


class CrossDomainLightningModule(
    models.CrossDomainModel, mixins.MeanMetricsMixin, pl.LightningModule
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def get_metrics_and_loss(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        (x, y), task_description = batch
        logits = self.forward(x, task_description)
        loss_func = get_loss_func(
            task_description.task_target, task_description.classes, self.device
        )
        pred_func = get_prob_func(task_description.task_target)
        loss = loss_func(logits, y)
        pred = pred_func(logits)
        return self.compute_metrics(task_description, pred, y), loss

    def compute_metrics(self, task_description, pred, y):
        raise NotImplementedError

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
