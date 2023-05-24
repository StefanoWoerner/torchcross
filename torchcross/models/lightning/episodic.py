from typing import Sequence

import torch
from lightning import pytorch as pl
from torch import Tensor

from torchcross import models
from torchcross.data.task import Task
from torchcross.models.lightning import mixins


class EpisodicLightningModule(
    models.EpisodicModel, mixins.MeanMetricsMixin, pl.LightningModule
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

    def get_metrics_and_losses(
        self, batch: Sequence[Task]
    ) -> tuple[list[tuple[Tensor, ...]], list[Tensor], list[Tensor]]:
        return [], [], self.forward(batch)

    def training_step(self, batch: Sequence[Task], batch_idx: int):
        self.optimizers().zero_grad()

        metric_values, support_losses, query_losses = self.get_metrics_and_losses(batch)

        self.update_metrics("train", metric_values)
        support_loss = torch.stack(support_losses).mean()
        query_loss = torch.stack(query_losses).mean()

        self.manual_backward(query_loss)
        self.optimizers().step()

        self.log("loss/metatrain/support", support_loss, batch_size=len(batch))
        self.log(
            "loss/metatrain/query", query_loss, batch_size=len(batch), prog_bar=True
        )
        self.log_dict(self.training_metrics)

    def validation_step(self, batch: Sequence[Task], batch_idx: int):
        torch.set_grad_enabled(True)
        metric_values, support_losses, query_losses = self.get_metrics_and_losses(batch)
        torch.set_grad_enabled(False)

        self.update_metrics("val", metric_values)
        support_loss = torch.stack(support_losses).mean()
        query_loss = torch.stack(query_losses).mean()

        self.log("loss/metaval/support", support_loss, batch_size=len(batch))
        self.log("loss/metaval/query", query_loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.validation_metrics)

    def test_step(self, batch: Sequence[Task], batch_idx: int):
        torch.set_grad_enabled(True)
        metric_values, support_losses, query_losses = self.get_metrics_and_losses(batch)
        torch.set_grad_enabled(False)

        self.update_metrics("test", metric_values)
        support_loss = torch.stack(support_losses).mean()
        query_loss = torch.stack(query_losses).mean()

        self.log("loss/metatest/support", support_loss, batch_size=len(batch))
        self.log(
            "loss/metatest/query", query_loss, batch_size=len(batch), prog_bar=True
        )
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        return self.outer_optimizer(self.parameters())
