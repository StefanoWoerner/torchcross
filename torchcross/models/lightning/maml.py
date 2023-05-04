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


class MAML(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optim_cfg,
        hparams: DictConfig,
        num_classes,
        inductive=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = hparams
        self.save_hyperparameters(hparams)

        self.net = net

        self.optim_cfg = optim_cfg

        self.inductive = inductive

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def outer_loop(
        self, batch: Sequence[Task], train: bool, metrics: dict[str, Metric]
    ):
        inner_optimizer = self.optim_cfg.inner_optimizer(self.net)
        # inner_optimizer = torchopt.MetaSGD(self.net, lr=1e-1)

        mode = "metatrain" if train else "metaval"

        support_losses = []
        query_losses = []

        net_state_dict = torchopt.extract_state_dict(
            self.net, by="reference", detach_buffers=True
        )
        optim_state_dict = torchopt.extract_state_dict(inner_optimizer, by="reference")

        for task in batch:
            support_x, support_y = task.support
            query_x, query_y = task.query

            if support_x.shape[1] == 1:
                support_x = support_x.repeat(1, 3, 1, 1)
                query_x = query_x.repeat(1, 3, 1, 1)

            self.net.train()

            for k in range(self.cfg.num_inner_steps):
                support_logit = self.net(support_x)
                loss = F.cross_entropy(support_logit, support_y)
                inner_optimizer.step(loss)

            if self.inductive:
                self.net.eval()

            with torch.no_grad():
                support_logit = self.net(support_x)
                support_losses.append(F.cross_entropy(support_logit, support_y))
                support_pred = torch.softmax(support_logit, dim=-1)
                metrics["support_accuracy"](support_pred, support_y)

            query_logit = self.net(query_x)
            query_losses.append(F.cross_entropy(query_logit, query_y))
            with torch.no_grad():
                query_pred = torch.softmax(query_logit, dim=-1)
                metrics["query_accuracy"](query_pred, query_y)

            torchopt.recover_state_dict(self.net, net_state_dict)
            torchopt.recover_state_dict(inner_optimizer, optim_state_dict)

        support_loss = sum(support_losses) / len(support_losses)
        query_loss = sum(query_losses) / len(query_losses)
        self.log(f"loss/{mode}/support", support_loss, batch_size=len(batch))
        self.log(f"loss/{mode}/query", query_loss, batch_size=len(batch), prog_bar=True)

        return query_loss

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

        # for pn, pp in self.named_parameters():
        #     self.logger.experiment.add_histogram(
        #         "parameters/" + pn, pp, self.global_step
        #     )
        # for pn, pp in self.named_parameters():
        #     self.logger.experiment.add_histogram(
        #         "gradients/" + pn, pp.grad, self.global_step
        #     )

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
