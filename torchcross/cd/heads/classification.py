from typing import Literal

import torch

from torchcross.data.task import TaskTarget


def new_classification_head(
    task_target: TaskTarget,
    classes: dict[int, str],
    num_in_features: int,
    device: torch.device = None,
    init: Literal["default", "kaiming", "zero"] = "zero",
) -> torch.nn.Linear:
    """Create a new classification head for a given task target.

    Args:
        task_target: The task target.
        classes: The classes for the task.
        num_in_features: The number of input features for the Linear layer.
        device: The device to create the Linear layer on.
        init: The initialization method for the Linear layer.

    Returns:
        A new Linear layer for the given task target and classes.
    """
    match task_target:
        case (
            TaskTarget.MULTICLASS_CLASSIFICATION
            | TaskTarget.MULTILABEL_CLASSIFICATION
            | TaskTarget.ORDINAL_REGRESSION
        ):
            fc = torch.nn.Linear(num_in_features, len(classes), device=device)
        case TaskTarget.BINARY_CLASSIFICATION | TaskTarget.REGRESSION:
            fc = torch.nn.Linear(num_in_features, 1, device=device)
        case target:
            raise NotImplementedError(f"Task target {target} is not supported")
    if init == "zero":
        torch.nn.init.zeros_(fc.weight.data)
        torch.nn.init.zeros_(fc.bias.data)
    elif init == "kaiming" or init == "default":
        pass
    else:
        raise ValueError(f"Unknown init method {init}")
    return fc
