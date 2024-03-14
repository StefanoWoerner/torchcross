from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from torchcross.data.task import TaskTarget, TaskDescription

__all__ = ["get_loss_func"]


def get_loss_func(
    task_target: TaskTarget,
    classes: dict[int, str],
    device: torch.device = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get the loss function for a given task target and classes.

    Args:
        task_target: The task target.
        classes: The classes for the task.
        device: The device to create tensors needed for the loss function
            on.

    Returns:
        A loss function that takes logits and labels and returns a loss
        tensor.
    """
    match task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            return lambda p, y, pos_class_weights=None: F.cross_entropy(
                p, y, weight=pos_class_weights
            )
        case TaskTarget.MULTILABEL_CLASSIFICATION | TaskTarget.BINARY_CLASSIFICATION:
            return (
                lambda p, y, pos_class_weights=None: F.binary_cross_entropy_with_logits(
                    p, y.float(), pos_weight=pos_class_weights
                )
            )
        case TaskTarget.ORDINAL_REGRESSION:
            class_to_idx = torch.full(
                (max(classes) + 1,),
                -1,
                dtype=torch.long,
                device=device,
            )
            for i, c in enumerate(classes):
                class_to_idx[c] = i
            return lambda p, y, pos_class_weights=None: F.cross_entropy(
                p, class_to_idx[y], weight=pos_class_weights
            )
        case TaskTarget.REGRESSION:
            return F.mse_loss
        case target:
            raise NotImplementedError(f"Unsupported task target: {target}")


class IndexedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        classes: dict[int, str],
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )

        self.register_buffer(
            "class_to_idx", torch.full((max(classes) + 1,), -1, dtype=torch.long)
        )
        for i, c in enumerate(classes):
            self.class_to_idx[c] = i

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, self.class_to_idx[target])


class FloatBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target.float())


def get_criterion(
    task_description: TaskDescription,
    pos_class_weights: torch.Tensor = None,
    device: torch.device = None,
    **criterion_kwargs,
):
    match task_description.task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            return nn.CrossEntropyLoss(weight=pos_class_weights, **criterion_kwargs)
        case TaskTarget.MULTILABEL_CLASSIFICATION | TaskTarget.BINARY_CLASSIFICATION:
            return FloatBCEWithLogitsLoss(
                pos_weight=pos_class_weights, **criterion_kwargs
            )
        case TaskTarget.ORDINAL_REGRESSION:
            return IndexedCrossEntropyLoss(
                task_description.classes, weight=pos_class_weights, **criterion_kwargs
            ).to(device)
        case TaskTarget.REGRESSION:
            return nn.MSELoss(**criterion_kwargs)
        case target:
            raise NotImplementedError(f"Unsupported task target: {target}")


def get_ordinal_cross_entropy_loss(classes):
    # Compute the weights for ordinal cross-entropy loss
    # using the intervals between consecutive class keys.
    sorted_class_keys = sorted(classes.keys())
    sorted_keys_tensor = torch.tensor(sorted_class_keys, dtype=torch.float)
    label_intervals_tensor = torch.diff(sorted_keys_tensor)
    weights = torch.cat((label_intervals_tensor, torch.tensor([1])), dim=0)

    def ordinal_cross_entropy(p, y):
        y = y.to(torch.long).unsqueeze(1)
        y_binary = y <= torch.arange(p.size(1), device=y.device)

        weightss = weights.unsqueeze(0).repeat(y.size(0), 1).to(p.device)

        binary_losses = F.binary_cross_entropy_with_logits(p, y_binary.float())
        weighted_binary_losses = binary_losses * weightss
        ordinal_loss = weighted_binary_losses.mean()
        return ordinal_loss

    return ordinal_cross_entropy
