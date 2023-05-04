import logging
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torchmetrics
from omegaconf import DictConfig

from torchcross.cd.heads import new_classification_head
from torchcross.cd.metrics import get_accuracy_func, get_auroc_func
from torchcross.cd.losses import get_loss_func
from torchcross.data.task import TaskTarget, TaskDescription
from torchcross.models.lightning import Classifier


class DiverseTransfer(Classifier):
    def __init__(
        self,
        backbone: tuple[torch.nn.Module, int],
        optim_cfg,
        hparams: DictConfig,
        *args,
        **kwargs,
    ) -> None:
        super(Classifier, self).__init__(*args, **kwargs)

        self.cfg = hparams
        self.save_hyperparameters(hparams)

        self.backbone, self.num_backbone_features = backbone

        self.optim_cfg = optim_cfg

        self.train_accuracy = torchmetrics.MeanMetric()
        self.train_AUROC = torchmetrics.MeanMetric()
        self.val_accuracy = torchmetrics.MeanMetric()
        self.val_AUROC = torchmetrics.MeanMetric()
        self.test_accuracy = torchmetrics.MeanMetric()
        self.test_AUROC = torchmetrics.MeanMetric()

        self.heads = torch.nn.ModuleDict()

    def forward(
        self, x: torch.Tensor, task_description: TaskDescription
    ) -> torch.Tensor:
        head = self.get_head(task_description, create=True, device=x.device)
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

        head = self.get_head(batch_task_description, create=True, device=x.device)
        loss_func = self.get_loss_func(batch_task_description, device=y.device)
        pred_func = self.get_pred_func(batch_task_description)
        accuracy_func, auroc_func = self.get_metric_func(
            batch_task_description, device=y.device
        )
        net = torch.nn.Sequential(OrderedDict(backbone=self.backbone, head=head))

        logits = net(x)
        with torch.no_grad():
            pred = pred_func(logits)
            self.train_accuracy(accuracy_func(pred, y))
            self.train_AUROC(auroc_func(pred, y))
        loss = loss_func(logits, y)

        self.log("loss/train", loss, batch_size=len(y))
        self.log_dict(
            {
                "accuracy/train": self.train_accuracy,
                "AUROC/train": self.train_AUROC,
            },
            prog_bar=True,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        (x, y), batch_task_description = batch

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if len(self.heads) == 0:
            logging.info("No heads found. Skipping validation step.")
            return

        loss_func = self.get_loss_func(batch_task_description, device=y.device)
        pred_func = self.get_pred_func(batch_task_description)
        accuracy_func, auroc_func = self.get_metric_func(
            batch_task_description, device=y.device
        )

        logits = self.forward(x, batch_task_description)
        pred = pred_func(logits)
        self.val_accuracy(accuracy_func(pred, y))
        self.val_AUROC(auroc_func(pred, y))
        loss = loss_func(logits, y)

        self.log("loss/val", loss, batch_size=len(y))
        self.log_dict(
            {
                "accuracy/val": self.val_accuracy,
                "AUROC/val": self.val_AUROC,
            },
            prog_bar=True,
        )

    def test_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], TaskDescription],
        batch_idx: int,
    ):
        (x, y), batch_task_description = batch

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        loss_func = self.get_loss_func(batch_task_description, device=y.device)
        pred_func = self.get_pred_func(batch_task_description)
        accuracy_func, auroc_func = self.get_metric_func(
            batch_task_description, device=y.device
        )

        logits = self.forward(x, batch_task_description)
        pred = pred_func(logits)
        self.test_accuracy(accuracy_func(pred, y))
        self.test_AUROC(auroc_func(pred, y))
        loss = loss_func(logits, y)

        self.log("loss/test", loss, batch_size=len(y))
        self.log_dict(
            {
                "accuracy/test": self.test_accuracy,
                "AUROC/test": self.test_AUROC,
            },
            prog_bar=True,
        )

    def get_head(
        self, task_description: TaskDescription, create=False, device=None
    ) -> torch.nn.Module:
        task_identifier = task_description.task_identifier
        if task_identifier not in self.heads:
            if create:
                if device is None:
                    raise ValueError("Device must be specified if creating head.")
                new_head = new_classification_head(
                    task_description.task_target,
                    task_description.classes,
                    self.num_backbone_features,
                    device,
                )
                self.heads[task_identifier] = new_head
                self.optimizers().add_param_group({"params": new_head.parameters()})
            else:
                raise ValueError(f"Head for task {task_identifier} not found.")
        return self.heads[task_identifier]

    def get_loss_func(
        self, task_description: TaskDescription, device=None
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return get_loss_func(
            task_description.task_target, task_description.classes, device
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

    def get_metric_func(
        self, task_description: TaskDescription, device=None
    ) -> tuple[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        return (
            get_accuracy_func(
                task_description.task_target, task_description.classes, device
            ),
            get_auroc_func(
                task_description.task_target, task_description.classes, device
            ),
        )
