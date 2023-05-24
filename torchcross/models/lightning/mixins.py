from typing import Literal

import torch
import torchmetrics
from torch import Tensor
import pytorch_lightning as pl


class MeanMetricsMixin(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.metric_keys = []
        self.training_metrics = torch.nn.ModuleDict()
        self.validation_metrics = torch.nn.ModuleDict()
        self.test_metrics = torch.nn.ModuleDict()

    def configure_metrics(self, metric_keys: list[str]) -> None:
        self.metric_keys = metric_keys
        self.training_metrics = torch.nn.ModuleDict({
            k.format("train"): torchmetrics.MeanMetric() for k in self.metric_keys
        })
        self.validation_metrics = torch.nn.ModuleDict({
            k.format("val"): torchmetrics.MeanMetric() for k in self.metric_keys
        })
        self.test_metrics = torch.nn.ModuleDict({
            k.format("test"): torchmetrics.MeanMetric() for k in self.metric_keys
        })

    def update_metrics(
        self,
        mode: Literal["train", "val", "test"],
        batch_metrics: list[tuple[Tensor, ...]] | tuple[Tensor, ...],
    ):
        metrics: torch.nn.ModuleDict
        if mode == "train":
            metrics = self.training_metrics
        elif mode == "val":
            metrics = self.validation_metrics
        elif mode == "test":
            metrics = self.test_metrics
        else:
            raise ValueError(f"Unknown mode {mode}")

        match batch_metrics:
            case tuple():
                aggregated = batch_metrics
                batch_size = 1
            case list():
                aggregated = [torch.stack(bm).mean() for bm in zip(*batch_metrics)]
                batch_size = len(batch_metrics)
            case _:
                raise ValueError(f"Unknown batch_metrics type {type(batch_metrics)}")

        for k, v in zip(self.metric_keys, aggregated):
            metrics[k.format(mode)](v, batch_size)
