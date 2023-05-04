from collections import OrderedDict
from functools import partial
from typing import Sequence, Callable

import torch
import torch.nn.functional as F
import torchmetrics
import torchopt
from omegaconf import DictConfig
from torchmetrics.metric import Metric

from torchcross.cd.heads import new_classification_head
from torchcross.cd.losses import get_loss_func
from torchcross.cd.metrics import get_accuracy_func
from torchcross.data.task import Task, TaskTarget
from torchcross.models.lightning import MAML


class DiverseMAML(MAML):
    def __init__(
        self,
        backbone: tuple[torch.nn.Module, int],
        # head_init: Literal['kaiming', 'zero', 'adaptive', 'unicorn', 'proto'] = 'zero',
        optim_cfg,
        hparams: DictConfig,
        inductive=True,
        *args,
        **kwargs,
    ) -> None:
        super(MAML, self).__init__(*args, **kwargs)

        self.cfg = hparams
        self.save_hyperparameters(hparams)

        self.backbone, self.num_backbone_features = backbone

        self.optim_cfg = optim_cfg

        self.inductive = inductive

        self.automatic_optimization = False

        self.train_support_accuracy = torchmetrics.MeanMetric()
        self.train_query_accuracy = torchmetrics.MeanMetric()
        self.val_support_accuracy = torchmetrics.MeanMetric()
        self.val_query_accuracy = torchmetrics.MeanMetric()
        self.test_support_accuracy = torchmetrics.MeanMetric()
        self.test_query_accuracy = torchmetrics.MeanMetric()

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

            inner_optimizer = self.optim_cfg.inner_optimizer(net)
            optim_state_dict = torchopt.extract_state_dict(
                inner_optimizer, by="reference"
            )

            net.train()

            for k in range(self.cfg.num_inner_steps):
                support_logit = net(support_x)
                loss = loss_func(support_logit, support_y)
                inner_optimizer.step(loss)

            if self.inductive:
                net.eval()

            with torch.no_grad():
                support_logit = net(support_x)
                support_losses.append(loss_func(support_logit, support_y))
                support_pred = pred_func(support_logit)
                support_accuracies.append(metric_func(support_pred, support_y))

            query_logit = net(query_x)
            query_losses.append(loss_func(query_logit, query_y))
            with torch.no_grad():
                query_pred = pred_func(query_logit)
                query_accuracies.append(metric_func(query_pred, query_y))

            torchopt.recover_state_dict(self.backbone, backbone_state_dict)
            # TODO: find out if the following line is even needed since the optimizer is only used for this single task
            torchopt.recover_state_dict(inner_optimizer, optim_state_dict)

        support_loss = sum(support_losses) / len(support_losses)
        query_loss = sum(query_losses) / len(query_losses)

        support_accuracy = sum(support_accuracies) / len(support_accuracies)
        query_accuracy = sum(query_accuracies) / len(query_accuracies)

        self.log(f"loss/{mode}/support", support_loss, batch_size=len(batch))
        self.log(f"loss/{mode}/query", query_loss, batch_size=len(batch), prog_bar=True)

        metrics["support_accuracy"](support_accuracy)
        metrics["query_accuracy"](query_accuracy)

        return query_loss

    def get_head(self, task: Task) -> torch.nn.Module:
        return new_classification_head(
            task.task_target,
            task.classes,
            self.num_backbone_features,
            task.support[0].device,
        )

    def get_loss_func(
        self, task: Task
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return get_loss_func(task.task_target, task.classes, task.support[0].device)

    def get_pred_func(self, task: Task) -> Callable[[torch.Tensor], torch.Tensor]:
        match task:
            case Task(task_target=TaskTarget.MULTICLASS_CLASSIFICATION):
                return partial(torch.softmax, dim=-1)
            case Task(
                task_target=(
                    TaskTarget.MULTILABEL_CLASSIFICATION
                    | TaskTarget.BINARY_CLASSIFICATION
                )
            ):
                return torch.sigmoid
            case Task(task_target=(TaskTarget.ORDINAL_REGRESSION)):
                return partial(torch.softmax, dim=-1)
            case Task(task_target):
                raise NotImplementedError(
                    f"Task target {task_target} not yet implemented"
                )

    def get_metric_func(
        self, task: Task
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return get_accuracy_func(task.task_target, task.classes, task.support[0].device)
