from functools import partial
from typing import Callable, Any, Optional, Literal

import torch
import torchmetrics
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAUROC,
    BinaryAUROC,
    MultilabelAUROC,
    MulticlassAccuracy,
    BinaryAccuracy,
    MultilabelAccuracy,
    Accuracy as _Accuracy,
    AUROC as _AUROC,
)

from torchcross.data.task import TaskTarget, TaskDescription


def get_accuracy_func(
    task_target: TaskTarget, classes: dict[int, str], device: torch.device = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get the accuracy function for a given task target and classes.

    Args:
        task_target: The task target.
        classes: The classes for the task.
        device: The device to create tensors needed for the accuracy
            function on.

    Returns:
        A torchmetrics functional metric or a function wrapping a
        torchmetrics functional metric for the given task target and
        classes.
    """
    match task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            return partial(
                torchmetrics.functional.classification.multiclass_accuracy,
                num_classes=len(classes),
                average="micro",
            )
        case TaskTarget.MULTILABEL_CLASSIFICATION:
            return partial(
                torchmetrics.functional.classification.multilabel_accuracy,
                num_labels=len(classes),
                average="micro",
            )
        case TaskTarget.BINARY_CLASSIFICATION:
            return torchmetrics.functional.classification.binary_accuracy
        case TaskTarget.ORDINAL_REGRESSION:
            class_to_idx = torch.full(
                (max(classes) + 1,),
                -1,
                dtype=torch.long,
                device=device,
            )
            for i, c in enumerate(classes):
                class_to_idx[c] = i
            acc = partial(
                torchmetrics.functional.classification.multiclass_accuracy,
                num_classes=len(classes),
                average="micro",
            )
            return lambda p, y: acc(p, class_to_idx[y])
        case target:
            raise NotImplementedError(f"Unsupported task target: {target}")


def get_auroc_func(
    task_target: TaskTarget, classes: dict[int, str], device: torch.device = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get the area under the ROC curve metric function for a given task
    target and classes.

    Args:
        task_target: The task target.
        classes: The classes for the task.
        device: The device to create tensors needed for the metric
            function on.

    Returns:
        A torchmetrics functional metric or a function wrapping a
        torchmetrics functional metric for the given task target and
        classes.
    """
    match task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            return partial(
                torchmetrics.functional.classification.multiclass_auroc,
                num_classes=len(classes),
            )
        case TaskTarget.MULTILABEL_CLASSIFICATION:
            return partial(
                torchmetrics.functional.classification.multilabel_auroc,
                num_labels=len(classes),
            )
        case TaskTarget.BINARY_CLASSIFICATION:
            return torchmetrics.functional.classification.binary_auroc
        case TaskTarget.ORDINAL_REGRESSION:
            class_to_idx = torch.full(
                (max(classes) + 1,),
                -1,
                dtype=torch.long,
                device=device,
            )
            for i, c in enumerate(classes):
                class_to_idx[c] = i
            auroc = partial(
                torchmetrics.functional.classification.multiclass_auroc,
                num_classes=len(classes),
            )
            return lambda p, y: auroc(p, class_to_idx[y])
        case target:
            raise NotImplementedError(f"Unsupported task target: {target}")


class IndexedMulticlassAccuracy(MulticlassAccuracy):
    def __init__(
        self,
        classes: dict[int, str],
        top_k: Optional[int] = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
    ) -> None:
        super().__init__(
            len(classes),
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
        )
        self.register_buffer("class_to_idx", torch.full(
            (max(classes) + 1,), -1, dtype=torch.long, device=self.device
        ))
        for i, c in enumerate(classes):
            self.class_to_idx[c] = i

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(preds, self.class_to_idx[target])


class Accuracy(_Accuracy):
    def __new__(
        cls,
        task_description: TaskDescription,
        *,
        top_k: int = 1,
        threshold: float = 0.5,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        kwargs.update(
            dict(
                multidim_average=multidim_average,
                ignore_index=ignore_index,
                validate_args=validate_args,
            )
        )

        task_target = task_description.task_target
        classes = task_description.classes

        if task_target == TaskTarget.MULTICLASS_CLASSIFICATION:
            assert isinstance(classes, dict)
            assert isinstance(top_k, int)
            return MulticlassAccuracy(len(classes), top_k, average, **kwargs)
        if task_target == TaskTarget.MULTILABEL_CLASSIFICATION:
            assert isinstance(classes, dict)
            return MultilabelAccuracy(len(classes), threshold, average, **kwargs)
        if task_target == TaskTarget.BINARY_CLASSIFICATION:
            return BinaryAccuracy(threshold, **kwargs)
        if task_target == TaskTarget.ORDINAL_REGRESSION:
            assert isinstance(classes, dict)
            return IndexedMulticlassAccuracy(classes, top_k, average, **kwargs)
        if task_target == TaskTarget.REGRESSION:
            raise ValueError("Accuracy metric not defined for regression tasks")
        raise NotImplementedError(f"Unsupported task target: {task_target}")


class IndexedMulticlassAUROC(MulticlassAUROC):
    def __init__(
        self,
        classes: dict[int, str],
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        thresholds: Optional[int | list[float] | torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
    ) -> None:
        super().__init__(
            len(classes),
            average=average,
            thresholds=thresholds,
            ignore_index=ignore_index,
            validate_args=validate_args,
        )
        self.register_buffer("class_to_idx", torch.full(
            (max(classes) + 1,), -1, dtype=torch.long, device=self.device
        ))
        for i, c in enumerate(classes):
            self.class_to_idx[c] = i

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        super().update(preds, self.class_to_idx[target])


class AUROC(_AUROC):
    def __new__(
        cls,
        task_description: TaskDescription,
        *,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        max_fpr: Optional[float] = None,
        thresholds: Optional[int | list[float] | torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        kwargs.update(
            dict(
                thresholds=thresholds,
                ignore_index=ignore_index,
                validate_args=validate_args,
            )
        )

        task_target = task_description.task_target
        classes = task_description.classes

        if task_target == TaskTarget.MULTICLASS_CLASSIFICATION:
            assert isinstance(classes, dict)
            return MulticlassAUROC(len(classes), average, **kwargs)
        if task_target == TaskTarget.MULTILABEL_CLASSIFICATION:
            assert isinstance(classes, dict)
            return MultilabelAUROC(len(classes), average, **kwargs)
        if task_target == TaskTarget.BINARY_CLASSIFICATION:
            return BinaryAUROC(max_fpr, **kwargs)
        if task_target == TaskTarget.ORDINAL_REGRESSION:
            assert isinstance(classes, dict)
            return IndexedMulticlassAUROC(classes, average, **kwargs)
        if task_target == TaskTarget.REGRESSION:
            raise ValueError("AUROC metric not defined for regression tasks")
        raise NotImplementedError(f"Unsupported task target: {task_target}")
