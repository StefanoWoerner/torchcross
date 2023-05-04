from typing import Callable

import torch
import torch.nn.functional as F

from torchcross.data.task import TaskTarget


def get_loss_func(
    task_target: TaskTarget, classes: dict[int, str], device: torch.device = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    match task_target:
        case TaskTarget.MULTICLASS_CLASSIFICATION:
            return F.cross_entropy
        case TaskTarget.MULTILABEL_CLASSIFICATION | TaskTarget.BINARY_CLASSIFICATION:
            return lambda p, y: F.binary_cross_entropy_with_logits(p, y.float())
        case TaskTarget.ORDINAL_REGRESSION:
            class_to_idx = torch.full(
                (max(classes) + 1,),
                -1,
                dtype=torch.long,
                device=device,
            )
            for i, c in enumerate(classes):
                class_to_idx[c] = i
            return lambda p, y: F.cross_entropy(p, class_to_idx[y])
        case TaskTarget.REGRESSION:
            return F.mse_loss
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

        binary_losses = F.binary_cross_entropy_with_logits(
            p, y_binary.float()
        )
        weighted_binary_losses = binary_losses * weightss
        ordinal_loss = weighted_binary_losses.mean()
        return ordinal_loss

    return ordinal_cross_entropy
