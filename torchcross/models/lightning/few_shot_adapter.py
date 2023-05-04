from collections import OrderedDict
from functools import partial
from typing import Sequence, Callable

import torch
import torch.nn.functional as F
import torchmetrics
import torchopt
import lightning.pytorch as pl
from omegaconf import DictConfig

from torchcross.cd.heads import new_classification_head
from torchcross.cd.losses import get_loss_func
from torchcross.cd.metrics import get_accuracy_func, get_auroc_func
from torchcross.data.task import Task, TaskTarget


class FewShotAdapter(pl.LightningModule):
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
        super().__init__(*args, **kwargs)

        self.cfg = hparams
        self.save_hyperparameters(hparams)

        self.backbone, self.num_backbone_features = backbone

        self.optim_cfg = optim_cfg

        self.inductive = inductive

        self.automatic_optimization = False

        self.support_accuracy = torchmetrics.MeanMetric()
        self.query_accuracy = torchmetrics.MeanMetric()
        self.support_accuracy_list = torchmetrics.CatMetric()
        self.query_accuracy_list = torchmetrics.CatMetric()

        self.support_auroc = torchmetrics.MeanMetric()
        self.query_auroc = torchmetrics.MeanMetric()
        self.support_auroc_list = torchmetrics.CatMetric()
        self.query_auroc_list = torchmetrics.CatMetric()

        self.logged_values = []

    def test_step(self, batch: list[Task], batch_idx: int):
        torch.set_grad_enabled(True)
        self.train()
        self.outer_loop(batch)
        torch.set_grad_enabled(False)

        self.log(
            "accuracy/metatest/support",
            self.support_accuracy,
            prog_bar=False,
        )
        self.log(
            "accuracy/metatest/query",
            self.query_accuracy,
            prog_bar=False,
        )
        self.log(
            "auroc/metatest/support",
            self.support_auroc,
            prog_bar=False,
        )
        self.log(
            "auroc/metatest/query",
            self.query_auroc,
            prog_bar=False,
        )

    # def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, list[EPOCH_OUTPUT]]) -> None:
    # self.logger.experiment.add_text(
    #     "accuracy/metatest/support_list",
    #     str(self.support_accuracy_list.compute()),
    #     global_step=self.global_step,
    # )
    # self.logger.experiment.add_text(
    #     "accuracy/metatest/query_list",
    #     str(self.query_accuracy_list.compute()),
    #     global_step=self.global_step,
    # )
    # self.logger.experiment.add_text(
    #     "auroc/metatest/support_list",
    #     str(self.support_auroc_list.compute()),
    #     global_step=self.global_step,
    # )
    # self.logger.experiment.add_text(
    #     "auroc/metatest/query_list",
    #     str(self.query_auroc_list.compute()),
    #     global_step=self.global_step,
    # )

    # self.log(
    #     "accuracy/metatest/support_list",
    #     self.support_accuracy_list,
    #     prog_bar=False,
    # )
    # self.log(
    #     "accuracy/metatest/query_list",
    #     self.query_accuracy_list,
    #     prog_bar=False,
    # )
    # self.log(
    #     "auroc/metatest/support_list",
    #     self.support_auroc_list,
    #     prog_bar=False,
    # )
    # self.log(
    #     "auroc/metatest/query_list",
    #     self.query_auroc_list,
    #     prog_bar=False,
    # )

    def outer_loop(self, batch: Sequence[Task]):
        support_losses = []
        query_losses = []

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
            accuracy_func, auroc_func = self.get_metric_func(task)
            net = torch.nn.Sequential(OrderedDict(backbone=self.backbone, head=head))

            inner_optimizer = self.optim_cfg.inner_optimizer(net.parameters())
            net.train()

            for k in range(self.cfg.num_inner_steps):
                support_logit = net(support_x)
                loss = loss_func(support_logit, support_y)
                inner_optimizer.zero_grad()  # zero gradients
                loss.backward()  # backward
                inner_optimizer.step()  # step updates

            if self.inductive:
                net.eval()

            with torch.no_grad():
                support_logit = net(support_x)
                support_losses.append(loss_func(support_logit, support_y))
                support_pred = pred_func(support_logit)
                support_accuracy = accuracy_func(support_pred, support_y)
                support_auroc = auroc_func(support_pred, support_y)
                self.support_accuracy(support_accuracy)
                self.support_accuracy_list(support_accuracy)
                self.support_auroc(support_auroc)
                self.support_auroc_list(support_auroc)

            query_logit = net(query_x)
            query_losses.append(loss_func(query_logit, query_y))
            with torch.no_grad():
                query_pred = pred_func(query_logit)
                query_accuracy = accuracy_func(query_pred, query_y)
                query_auroc = auroc_func(query_pred, query_y)
                self.query_accuracy(query_accuracy)
                self.query_accuracy_list(query_accuracy)
                self.query_auroc(query_auroc)
                self.query_auroc_list(query_auroc)

            torchopt.recover_state_dict(self.backbone, backbone_state_dict)
            self.logged_values.append(
                {
                    "support_loss": support_losses[-1].item(),
                    "query_loss": query_losses[-1].item(),
                    "support_accuracy": support_accuracy.item(),
                    "query_accuracy": query_accuracy.item(),
                    "support_auroc": support_auroc.item(),
                    "query_auroc": query_auroc.item(),
                }
            )

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
    ) -> tuple[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        return (
            get_accuracy_func(task.task_target, task.classes, task.support[0].device),
            get_auroc_func(task.task_target, task.classes, task.support[0].device),
        )
