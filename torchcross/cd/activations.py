from functools import partial
from typing import Callable

import torch
from torch.nn import functional as F

from torchcross.data.task import TaskTarget

__all__ = ["get_prob_func", "get_log_prob_func"]


def get_prob_func(
    task_target: TaskTarget,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get the probability function for a given task target.

    Args:
        task_target: The task target.

    Returns:
        A function to convert logits to probabilities.
    """
    match task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            return partial(torch.softmax, dim=-1)
        case TaskTarget.MULTILABEL_CLASSIFICATION | TaskTarget.BINARY_CLASSIFICATION:
            return torch.sigmoid
        case TaskTarget.ORDINAL_REGRESSION:
            return partial(torch.softmax, dim=-1)
        case TaskTarget.REGRESSION:
            return lambda x: x
        case target:
            raise NotImplementedError(f"Task target {target} not yet implemented")


def get_log_prob_func(
    task_target: TaskTarget,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get the log probability function for a given task target.

    Args:
        task_target: The task target.

    Returns:
        A function to convert logits to log probabilities.
    """
    match task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            return partial(torch.log_softmax, dim=-1)
        case TaskTarget.MULTILABEL_CLASSIFICATION | TaskTarget.BINARY_CLASSIFICATION:
            return F.logsigmoid
        case TaskTarget.ORDINAL_REGRESSION:
            return partial(torch.log_softmax, dim=-1)
        case TaskTarget.REGRESSION:
            return lambda x: x
        case target:
            raise NotImplementedError(f"Task target {target} not yet implemented")
