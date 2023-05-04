import torch

from torchcross.data.task import TaskTarget


def new_classification_head(
    task_target: TaskTarget,
    classes: dict[int, str],
    num_in_features: int,
    device: torch.device,
) -> torch.nn.Module:
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
    fc.weight.data = torch.zeros_like(fc.weight.data)
    fc.bias.data = torch.zeros_like(fc.bias.data)
    return fc
