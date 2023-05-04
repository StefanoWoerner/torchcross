from functools import partial
from typing import Callable

import torch
import torchmetrics

from torchcross.data.task import TaskTarget


def get_accuracy_func(
    task_target: TaskTarget, classes: dict[int, str], device: torch.device = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Get the accuracy metric function for a given task target.
    :param task_target: the task target
    :param classes: a dictionary mapping class indices to class names
    :param device: a torch device (needed for ordinal regression metric)
    :return: a function expecting two tensors, the first being the
    predictions and the second being the ground truth, and returning
    the accuracy
    """
    match task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            return partial(
                torchmetrics.functional.classification.multiclass_accuracy,
                num_classes=len(classes),
            )
        case TaskTarget.MULTILABEL_CLASSIFICATION:
            return partial(
                torchmetrics.functional.classification.multilabel_accuracy,
                num_labels=len(classes),
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
            )
            return lambda p, y: acc(p, class_to_idx[y])
        case target:
            raise NotImplementedError(f"Unsupported task target: {target}")


def get_auroc_func(
    task_target: TaskTarget, classes: dict[int, str], device: torch.device = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Get the area under the ROC curve metric function for a given task
    target.
    :param task_target: the task target
    :param classes: a dictionary mapping class indices to class names
    :param device: a torch device (needed for ordinal regression metric)
    :return: a function expecting two tensors, the first being the
    predictions and the second being the ground truth, and returning
    the area under the ROC curve
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
