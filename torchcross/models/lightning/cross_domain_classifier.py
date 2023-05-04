import logging
from collections import OrderedDict
from functools import partial
from typing import Callable, Any

import lightning.pytorch as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch.optim import Optimizer

from torchcross.cd.heads import new_classification_head
from torchcross.cd.losses import get_loss_func
from torchcross.cd.metrics import get_accuracy_func, get_auroc_func
from torchcross.data.task import TaskTarget, TaskDescription
from torchcross.models.lightning import Classifier


class CrossDomainClassifier(Classifier):
    def __init__(
        self,
        backbone: tuple[torch.nn.Module, int],
        optimizer: Callable[[...], Optimizer],
        hparams: DictConfig,
        *args,
        **kwargs,
    ) -> None:
        super(Classifier, self).__init__(*args, **kwargs)
        self.save_hyperparameters(hparams)

        self.backbone, self.num_backbone_features = backbone
        self.optimizer = optimizer

        self.train_accuracy = torchmetrics.MeanMetric()
        self.train_AUROC = torchmetrics.MeanMetric()
        self.val_accuracy = torchmetrics.MeanMetric()
        self.val_AUROC = torchmetrics.MeanMetric()
        self.test_accuracy = torchmetrics.MeanMetric()
        self.test_AUROC = torchmetrics.MeanMetric()

        self.heads = torch.nn.ModuleDict()

    def forward(
        self,
        x: torch.Tensor,
        task_description: TaskDescription,
        create_head: bool = False,
    ) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        head = self.get_head(task_description, create=create_head)
        net = torch.nn.Sequential(OrderedDict(backbone=self.backbone, head=head))
        return net(x)

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        (x, y), batch_task_description = batch

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        head = self.get_head(batch_task_description, create=True)
        loss_func = self.get_loss_func(batch_task_description)
        pred_func = self.get_pred_func(batch_task_description)
        accuracy_func, auroc_func = self.get_metric_funcs(batch_task_description)
        net = torch.nn.Sequential(OrderedDict(backbone=self.backbone, head=head))

        logits = net(x)
        with torch.no_grad():
            pred = pred_func(logits)
            self.train_accuracy(accuracy_func(pred, y))
            self.train_AUROC(auroc_func(pred, y))
        loss = loss_func(logits, y)

        # self.log("loss/train", loss, batch_size=len(y))
        self.log_dict(
            {
                "loss/train": loss,
                "accuracy/train": self.train_accuracy,
                "AUROC/train": self.train_AUROC,
            },
            batch_size=len(y),
            prog_bar=True,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        if len(self.heads) == 0:
            logging.info("No heads found. Skipping validation step.")
            return
        loss, accuracy, auroc, batch_size = self._shared_eval_step(batch)
        self.val_accuracy(accuracy)
        self.val_AUROC(auroc)

        self.log_dict(
            {
                "loss/val": loss,
                "accuracy/val": self.val_accuracy,
                "AUROC/val": self.val_AUROC,
            },
            batch_size=batch_size,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        loss, accuracy, auroc, batch_size = self._shared_eval_step(batch)
        self.test_accuracy(accuracy)
        self.test_AUROC(auroc)

        self.log_dict(
            {
                "loss/test": loss,
                "accuracy/test": self.test_accuracy,
                "AUROC/test": self.test_AUROC,
            },
            batch_size=batch_size,
            prog_bar=True,
        )

    def _shared_eval_step(self, batch):
        (x, y), batch_task_description = batch
        loss_func = self.get_loss_func(batch_task_description)
        pred_func = self.get_pred_func(batch_task_description)
        accuracy_func, auroc_func = self.get_metric_funcs(batch_task_description)
        logits = self.forward(x, batch_task_description)
        pred = pred_func(logits)
        accuracy = accuracy_func(pred, y)
        auroc = auroc_func(pred, y)
        loss = loss_func(logits, y)
        return loss, accuracy, auroc, len(y)

    def predict_step(
        self,
        batch: tuple[torch.Tensor, TaskDescription],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        x, batch_task_description = batch
        pred_func = self.get_pred_func(batch_task_description)
        pred_to_label_func = self.get_pred_to_label_func(batch_task_description)

        logits = self.forward(x, batch_task_description)
        pred = pred_func(logits)
        return pred_to_label_func(pred)

    def get_head(
        self, task_description: TaskDescription, create=False
    ) -> torch.nn.Module:
        task_identifier = task_description.task_identifier
        if task_identifier not in self.heads:
            if create:
                new_head = self._get_head(task_description)
                self.heads[task_identifier] = new_head
                self.optimizers().add_param_group({"params": new_head.parameters()})
            else:
                raise ValueError(f"Head for task {task_identifier} not found.")
        return self.heads[task_identifier]

    def _get_head(self, task_description: TaskDescription) -> torch.nn.Module:
        return new_classification_head(
            task_description.task_target,
            task_description.classes,
            self.num_backbone_features,
            self.device,
        )

    def get_loss_func(
        self, task_description: TaskDescription
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return get_loss_func(
            task_description.task_target, task_description.classes, self.device
        )

    def get_pred_func(
        self, task_description: TaskDescription
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        match task_description:
            case TaskDescription(task_target=TaskTarget.MULTICLASS_CLASSIFICATION):
                return partial(torch.softmax, dim=-1)
            case TaskDescription(
                task_target=(
                    TaskTarget.MULTILABEL_CLASSIFICATION
                    | TaskTarget.BINARY_CLASSIFICATION
                )
            ):
                return torch.sigmoid
            case TaskDescription(task_target=(TaskTarget.ORDINAL_REGRESSION)):
                return partial(torch.softmax, dim=-1)
            case TaskDescription(task_target):
                raise NotImplementedError(
                    f"Task target {task_target} not yet implemented"
                )

    def get_pred_to_label_func(
        self, task_description: TaskDescription
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        match task_description:
            case TaskDescription(task_target=TaskTarget.MULTICLASS_CLASSIFICATION):
                return lambda x: x.argmax(dim=1)
            case TaskDescription(
                task_target=(
                    TaskTarget.MULTILABEL_CLASSIFICATION
                    | TaskTarget.BINARY_CLASSIFICATION
                )
            ):
                return lambda x: x > 0.5
            case TaskDescription(task_target=(TaskTarget.ORDINAL_REGRESSION)):
                class_keys = torch.tensor(
                    task_description.classes.keys(),
                    dtype=torch.long,
                    device=self.device,
                )
                return lambda x: class_keys[x.argmax(dim=1)]
            case TaskDescription(task_target):
                raise NotImplementedError(
                    f"Task target {task_target} not yet implemented"
                )

    def get_metric_funcs(
        self, task_description: TaskDescription
    ) -> tuple[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        return (
            get_accuracy_func(
                task_description.task_target, task_description.classes, self.device
            ),
            get_auroc_func(
                task_description.task_target, task_description.classes, self.device
            ),
        )

    def configure_optimizers(
        self,
    ) -> Optimizer | tuple[list[Optimizer], list[Any]]:
        optimizer = self.optimizer(params=self.parameters())

        if "lr_scheduler" in self.hparams:
            scheduler = self.hparams.lr_scheduler(optimizer)
            return [optimizer], [scheduler]

        return optimizer
