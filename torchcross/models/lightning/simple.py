from typing import Callable, Any, Literal

import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import MetricCollection, Metric

from torchcross import models
from torchcross.cd.activations import get_prob_func
from torchcross.cd.losses import get_loss_func, get_criterion
from torchcross.cd.metrics import get_accuracy_func, get_auroc_func, Accuracy, AUROC
from torchcross.data.task import TaskDescription
from torchcross.models.lightning import mixins


class SimpleClassifier(models.SimpleClassifier, pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        task_description: TaskDescription,
        optimizer: type[Optimizer],
        lr_scheduler: Callable[[Optimizer], Any] | None = None,
        expand_input_channels: bool = True,
        pos_class_weights: torch.Tensor = None,
    ) -> None:
        super().__init__(
            backbone, num_backbone_features, task_description, expand_input_channels
        )

        self.register_buffer("pos_class_weights", pos_class_weights)
        self.criterion = get_criterion(
            task_description, pos_class_weights=pos_class_weights
        )
        self.pred_func = get_prob_func(task_description.task_target)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.training_metrics = MetricCollection(
            {"accuracy": Accuracy(task_description), "AUROC": AUROC(task_description)},
            postfix="/train",
        )
        self.validation_metrics = self.training_metrics.clone(postfix="/val")
        self.test_metrics = self.training_metrics.clone(postfix="/test")

        self.automatic_optimization = False

    def do_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        mode: Literal["train", "val", "test"],
    ) -> Tensor:
        x, y = batch
        logits = self.forward(x)
        # loss = self.loss_func(logits, y, self.pos_class_weights)
        loss = self.criterion(logits, y)
        pred = self.pred_func(logits)
        self.update_metrics(mode, pred, y)
        return loss

    def update_metrics(
        self, mode: Literal["train", "val", "test"], pred: torch.Tensor, y: torch.Tensor
    ):
        metrics = self.get_metrics(mode)
        return metrics(pred, y)

    def get_metrics(
        self, mode: Literal["train", "val", "test"]
    ) -> Metric | MetricCollection:
        if mode == "train":
            return self.training_metrics
        elif mode == "val":
            return self.validation_metrics
        elif mode == "test":
            return self.test_metrics
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        self.optimizers().zero_grad()

        loss = self.do_step(batch, "train")

        self.manual_backward(loss)
        self.optimizers().step()

        self.log("loss/train", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.training_metrics)

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss = self.do_step(batch, "val")

        self.log("loss/val", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.validation_metrics, prog_bar=True)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss = self.do_step(batch, "test")

        self.log("loss/test", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
