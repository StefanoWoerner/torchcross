from pprint import pprint
from typing import Literal

import torch
from lightning import pytorch as pl
from torch import Tensor

from torchcross import models
from torchcross.cd.activations import get_prob_func
from torchcross.cd.losses import get_loss_func
from torchcross.data.task import TaskDescription
from torchcross.models.lightning import mixins


class CrossDomainLightningModule(models.CrossDomainModel, pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.scale_loss_by_batch_count = False
        self.train_counts = {}
        self.val_counts = {}
        self.test_counts = {}
        self.previous_train_counts = {}
        self.previous_val_counts = {}
        self.previous_test_counts = {}

        self.automatic_optimization = False
        self.debug = False

    def get_metrics_and_loss(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        mode: Literal["train", "val", "test"],
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        (x, y), task_description = batch
        logits = self.forward(x, task_description)
        loss_func = get_loss_func(
            task_description.task_target, task_description.classes, self.device
        )
        pred_func = get_prob_func(task_description.task_target)
        loss = loss_func(logits, y)

        pred = pred_func(logits)
        task_counter = getattr(self, f"{mode}_counts")
        task_counter[task_description.task_identifier] = (
            task_counter.get(task_description.task_identifier, 0) + 1
        )
        return self.call_metrics(mode, task_description, pred, y), loss

    def on_train_epoch_start(self) -> None:
        self.previous_train_counts = self.train_counts
        self.train_counts = {}

    def on_validation_epoch_start(self) -> None:
        self.previous_val_counts = self.val_counts
        self.val_counts = {}

    def on_test_epoch_start(self) -> None:
        self.previous_test_counts = self.test_counts
        self.test_counts = {}

    def on_train_epoch_end(self) -> None:
        if self.debug:
            print("Train counts:")
            pprint(self.train_counts)
            print(len(self.train_counts))
            print(sum(self.train_counts.values()))

    def on_validation_epoch_end(self) -> None:
        if self.debug:
            print("Val counts:")
            pprint(self.val_counts)
            print(len(self.val_counts))
            print(sum(self.val_counts.values()))

    def on_test_epoch_end(self) -> None:
        if self.debug:
            print("Test counts:")
            pprint(self.test_counts)
            print(len(self.test_counts))
            print(sum(self.test_counts.values()))

    def call_metrics(
        self,
        mode: Literal["train", "val", "test"],
        task_description: TaskDescription,
        pred: torch.Tensor,
        y: torch.Tensor,
    ):
        raise NotImplementedError

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        self.optimizers().zero_grad()

        metric_values, loss = self.get_metrics_and_loss(batch, "train")

        scaled_loss = loss
        if self.scale_loss_by_batch_count:
            # Scale the loss by the count of batches for this task in the previous
            # epoch compared to the task with the most batches in the previous epoch.
            # We use the previous epoch because the current epoch is not finished yet
            # and therefore the counts are not final.
            task_identifier = batch[1].task_identifier
            previous_task_count = self.previous_train_counts.get(task_identifier, 0)
            if previous_task_count:
                # If the task was not present in the previous epoch, we do not scale the
                # loss, as it is the first epoch or the first time we see this task.
                previous_max_count = max(self.previous_train_counts.values())
                scale_factor = previous_max_count / previous_task_count
                scaled_loss = loss * scale_factor

        self.manual_backward(scaled_loss)
        self.optimizers().step()

        self.log("loss/train", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.training_metrics)

    def validation_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        metric_values, loss = self.get_metrics_and_loss(batch, "val")

        self.log("loss/val", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.validation_metrics, prog_bar=True)

    def test_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        metric_values, loss = self.get_metrics_and_loss(batch, "test")

        self.log("loss/test", loss, batch_size=len(batch), prog_bar=True)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
