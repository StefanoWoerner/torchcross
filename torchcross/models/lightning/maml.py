from collections.abc import Callable, Sequence, Iterable
from typing import Optional, Any

import torch
import torchopt
from torch import Tensor
from torch.optim import Optimizer

from .episodic import EpisodicLightningModule
from ... import models
from ...cd.activations import get_prob_func
from ...cd.losses import get_loss_func
from ...cd.metrics import get_accuracy_func, get_auroc_func
from ...data import TaskDescription, Task


class MAML(models.MAML, EpisodicLightningModule):
    def __init__(
        self,
        backbone: tuple[torch.nn.Module, int],
        task_description: TaskDescription,
        outer_optimizer: Callable[
            [Iterable[Tensor] | Iterable[dict[str, Any]]], torch.optim.Optimizer
        ],
        inner_optimizer: Callable[[torch.nn.Module], torchopt.MetaOptimizer],
        eval_inner_optimizer: Callable[
            [Sequence[torch.nn.Parameter]], torchopt.Optimizer
        ],
        num_inner_steps: int,
        eval_num_inner_steps: Optional[int] = None,
        transductive: bool = True,
        reset_head: Optional[models.maml.head_init_t] = None,
        outer_lr_scheduler: Callable[[Optimizer], Any] | None = None,
    ) -> None:
        super(MAML, self).__init__(
            *backbone,
            task_description,
            inner_optimizer,
            eval_inner_optimizer,
            num_inner_steps,
            eval_num_inner_steps,
            transductive,
            reset_head,
        )

        self.outer_optimizer = outer_optimizer
        self.outer_lr_scheduler = outer_lr_scheduler
        metric_keys = [
            "accuracy/meta{}/support",
            "accuracy/meta{}/query",
            "AUROC/meta{}/support",
            "AUROC/meta{}/query",
        ]
        self.configure_metrics(metric_keys)

        self.accuracy_func = get_accuracy_func(
            task_description.task_target, task_description.classes, self.device
        )
        self.auroc_func = get_auroc_func(
            task_description.task_target, task_description.classes, self.device
        )

    def compute_metrics(self, task, support_pred, query_pred):
        support_y = task.support[1]
        query_y = task.query[1]
        support_accuracy = self.accuracy_func(support_pred, support_y)
        query_accuracy = self.accuracy_func(query_pred, query_y)
        support_auroc = self.auroc_func(support_pred, support_y)
        query_auroc = self.auroc_func(query_pred, query_y)
        return support_accuracy, query_accuracy, support_auroc, query_auroc

    def get_metrics_and_losses(
        self, batch: Sequence[Task]
    ) -> tuple[list[tuple[Tensor, ...]], list[Tensor], list[Tensor]]:
        net_state_dict = torchopt.extract_state_dict(
            self.net, by="reference", detach_buffers=True
        )
        metric_values = []
        support_losses = []
        query_losses = []
        for task in batch:
            support_x, support_y = task.support
            query_x, query_y = task.query

            loss_func = get_loss_func(task.task_target, task.classes, self.device)
            pred_func = get_prob_func(task.task_target)

            support_logit, query_logit = self.episode(task, with_support_output=True)
            query_losses.append(loss_func(query_logit, query_y))
            with torch.no_grad():
                support_losses.append(loss_func(support_logit, support_y))
                support_pred = pred_func(support_logit)
                query_pred = pred_func(query_logit)
                metric_values.append(
                    self.compute_metrics(task, support_pred, query_pred)
                )

            torchopt.recover_state_dict(self.net, net_state_dict)
        return metric_values, support_losses, query_losses


class CrossDomainMAML(models.CrossDomainMAML, EpisodicLightningModule):
    def __init__(
        self,
        backbone: tuple[torch.nn.Module, int],
        outer_optimizer: Callable[
            [Iterable[Tensor] | Iterable[dict[str, Any]]], torch.optim.Optimizer
        ],
        inner_optimizer: Callable[[torch.nn.Module], torchopt.MetaOptimizer],
        eval_inner_optimizer: Callable[
            [Sequence[torch.nn.Parameter]], torchopt.Optimizer
        ],
        num_inner_steps: int,
        eval_num_inner_steps: Optional[int] = None,
        transductive: bool = True,
        head_init: Optional[models.maml.head_init_t] = None,
        outer_lr_scheduler: Callable[[Optimizer], Any] | None = None,
    ) -> None:
        super(CrossDomainMAML, self).__init__(
            *backbone,
            inner_optimizer,
            eval_inner_optimizer,
            num_inner_steps,
            eval_num_inner_steps,
            transductive,
            head_init,
        )

        self.outer_optimizer = outer_optimizer
        self.outer_lr_scheduler = outer_lr_scheduler
        metric_keys = [
            "accuracy/meta{}/support",
            "accuracy/meta{}/query",
            "AUROC/meta{}/support",
            "AUROC/meta{}/query",
        ]
        self.configure_metrics(metric_keys)

    def compute_metrics(self, task, support_pred, query_pred):
        accuracy_func = get_accuracy_func(task.task_target, task.classes, self.device)
        auroc_func = get_auroc_func(task.task_target, task.classes, self.device)
        support_y = task.support[1]
        query_y = task.query[1]
        support_accuracy = accuracy_func(support_pred, support_y)
        query_accuracy = accuracy_func(query_pred, query_y)
        support_auroc = auroc_func(support_pred, support_y)
        query_auroc = auroc_func(query_pred, query_y)
        return support_accuracy, query_accuracy, support_auroc, query_auroc

    def get_metrics_and_losses(
        self, batch: Sequence[Task]
    ) -> tuple[list[tuple[Tensor, ...]], list[Tensor], list[Tensor]]:
        backbone_state_dict = torchopt.extract_state_dict(
            self.backbone, by="reference", detach_buffers=True
        )
        metric_values = []
        support_losses = []
        query_losses = []
        for task in batch:
            support_x, support_y = task.support
            query_x, query_y = task.query

            loss_func = get_loss_func(task.task_target, task.classes, self.device)
            pred_func = get_prob_func(task.task_target)

            support_logit, query_logit = self.episode(task, with_support_output=True)
            query_losses.append(loss_func(query_logit, query_y))
            with torch.no_grad():
                support_losses.append(loss_func(support_logit, support_y))
                support_pred = pred_func(support_logit)
                query_pred = pred_func(query_logit)
                metric_values.append(
                    self.compute_metrics(task, support_pred, query_pred)
                )

            torchopt.recover_state_dict(self.backbone, backbone_state_dict)
        return metric_values, support_losses, query_losses
