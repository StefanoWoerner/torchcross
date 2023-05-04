from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import lightning.pytorch as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch.optim import Optimizer
import torch.nn.functional as F


class Classifier(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optim_cfg,
        hparams: DictConfig,
        num_classes,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(hparams)

        self.optim_cfg = optim_cfg
        self.net = net

        self.train_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        )
        self.train_AUROC = torchmetrics.AUROC("multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.val_AUROC = torchmetrics.AUROC("multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(
            "multiclass", num_classes=num_classes
        )
        self.test_AUROC = torchmetrics.AUROC("multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        with torch.no_grad():
            pred = torch.softmax(logits, dim=-1)
            self.train_accuracy(pred, y)
            self.train_AUROC(pred, y)
        loss = F.cross_entropy(logits, y)

        self.log("loss/train", loss, batch_size=len(y))
        self.log_dict(
            {
                "accuracy/train": self.train_accuracy,
                "AUC/train": self.train_AUROC,
            },
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        pred = torch.softmax(logits, dim=-1)
        self.val_accuracy(pred, y)
        self.val_AUROC(pred, y)
        loss = F.cross_entropy(logits, y)

        self.log("loss/val", loss, batch_size=len(y))
        self.log_dict(
            {
                "accuracy/val": self.val_accuracy,
                "AUC/val": self.val_AUROC,
            },
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        pred = torch.softmax(logits, dim=-1)
        self.test_accuracy(pred, y)
        self.test_AUROC(pred, y)
        loss = F.cross_entropy(logits, y)

        self.log("loss/test", loss, batch_size=len(y))
        self.log_dict(
            {
                "accuracy/test": self.test_accuracy,
                "AUC/test": self.test_AUROC,
            },
            prog_bar=True,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        logits = self.forward(x)
        pred = torch.softmax(logits, dim=-1)
        return pred

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
