from collections import OrderedDict
from typing import Any, Sequence, Tuple, Union

import torchopt
import hydra
import lightning.pytorch as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch.optim import Optimizer
import torch.nn.functional as F
from torchmetrics.metric import Metric

from torchcross.data.task import Task


class EpisodicModel(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optim_cfg,
        hparams: DictConfig,
        num_classes,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = hparams
        self.save_hyperparameters(hparams)

        self.net = net

        self.optim_cfg = optim_cfg

        self.automatic_optimization = False

        self.train_support_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        )
        self.train_query_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        )
        self.val_support_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        )
        self.val_query_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        )
        self.test_support_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        )
        self.test_query_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        )

    def forward(self, batch) -> torch.Tensor:

        backbone_state_dict = torchopt.extract_state_dict(
            self.backbone, by="reference", detach_buffers=True
        )

        query_logits = []

        for task in batch:
            support_x, support_y = task.support
            query_x, query_y = task.query

            if support_x.shape[1] == 1:
                support_x = support_x.repeat(1, 3, 1, 1)
                query_x = query_x.repeat(1, 3, 1, 1)

            head = self.get_head(task)
            loss_func = self.get_loss_func(task)
            pred_func = self.get_pred_func(task)
            metric_func = self.get_metric_func(task)
            net = torch.nn.Sequential(OrderedDict(backbone=self.backbone, head=head))
            self.inner_loop(net, support_x, support_y, loss_func)
            query_logits.append(net(query_x))
            torchopt.recover_state_dict(self.backbone, backbone_state_dict)

    def outer_loop(
        self, batch: Sequence[Task], train: bool, metrics: dict[str, Metric]
    ):
        mode = "metatrain" if train else "metaval"

        support_losses = []
        query_losses = []
        support_accuracies = []
        query_accuracies = []

        backbone_state_dict = torchopt.extract_state_dict(
            self.backbone, by="reference", detach_buffers=True
        )

        for task in batch:
            sl, ql, sm, qm = self.episode(task)
            torchopt.recover_state_dict(self.backbone, backbone_state_dict)

        support_loss = sum(support_losses) / len(support_losses)
        query_loss = sum(query_losses) / len(query_losses)

        support_accuracy = sum(support_accuracies) / len(support_accuracies)
        query_accuracy = sum(query_accuracies) / len(query_accuracies)

        self.log(f"loss/{mode}/support", support_loss, batch_size=len(batch))
        self.log(f"loss/{mode}/query", query_loss, batch_size=len(batch), prog_bar=True)

        metrics["support_accuracy"](support_accuracy)
        metrics["query_accuracy"](query_accuracy)

        return support_loss, query_loss, support_metrics, query_metrics

    def episode(self, task):
        support_x, support_y = task.support
        query_x, query_y = task.query
        if support_x.shape[1] == 1:
            support_x = support_x.repeat(1, 3, 1, 1)
            query_x = query_x.repeat(1, 3, 1, 1)
        head = self.get_head(task)
        loss_func = self.get_loss_func(task)
        pred_func = self.get_pred_func(task)
        metric_func = self.get_metric_func(task)
        net = torch.nn.Sequential(OrderedDict(backbone=self.backbone, head=head))
        self.inner_loop(net, support_x, support_y, loss_func)
        with torch.no_grad():
            support_logit = net(support_x)
            support_loss = loss_func(support_logit, support_y)
            support_pred = pred_func(support_logit)
            support_metric = metric_func(support_pred, support_y)
        query_logit = net(query_x)
        query_loss = loss_func(query_logit, query_y)
        with torch.no_grad():
            query_pred = pred_func(query_logit)
            query_metric = metric_func(query_pred, query_y)
        return support_loss, query_loss, support_metric, query_metric

    def inner_loop(self, net, support_x, support_y, loss_func):
        inner_optimizer = self.optim_cfg.inner_optimizer(net)
        net.train()
        for k in range(self.cfg.num_inner_steps):
            support_logit = net(support_x)
            loss = loss_func(support_logit, support_y)
            inner_optimizer.step(loss)

    def training_step(self, batch: Any, batch_idx: int):
        outer_optimizer = self.optimizers()
        outer_optimizer.zero_grad()

        loss = self.outer_loop(
            batch,
            True,
            {
                "support_accuracy": self.train_support_accuracy,
                "query_accuracy": self.train_query_accuracy,
            },
        )

        self.manual_backward(loss)

        self.log(
            "accuracy/metatrain/support_accuracy",
            self.train_support_accuracy,
            prog_bar=False,
        )
        self.log(
            "accuracy/metatrain/query_accuracy",
            self.train_query_accuracy,
            prog_bar=True,
        )

        self.optimizers().step()

    def validation_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        self.train()
        self.outer_loop(
            batch,
            False,
            {
                "support_accuracy": self.val_support_accuracy,
                "query_accuracy": self.val_query_accuracy,
            },
        )
        torch.set_grad_enabled(False)

        self.log(
            "accuracy/metaval/support_accuracy",
            self.val_support_accuracy,
            prog_bar=False,
        )
        self.log(
            "accuracy/metaval/query_accuracy",
            self.val_query_accuracy,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        torch.set_grad_enabled(True)
        self.train()
        self.outer_loop(
            batch,
            False,
            {
                "support_accuracy": self.test_support_accuracy,
                "query_accuracy": self.test_query_accuracy,
            },
        )
        torch.set_grad_enabled(False)

        self.log(
            "accuracy/metaval/support_accuracy",
            self.test_support_accuracy,
            prog_bar=False,
        )
        self.log(
            "accuracy/metaval/query_accuracy",
            self.test_query_accuracy,
            prog_bar=True,
        )

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:

        outer_optimizer = self.optim_cfg.outer_optimizer(params=self.parameters())

        if self.optim_cfg.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(
                self.optim_cfg.lr_scheduler, optimizer=outer_optimizer
            )
            return [outer_optimizer], [scheduler]

        return outer_optimizer
