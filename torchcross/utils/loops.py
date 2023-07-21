from typing import Callable

import torch
import torchopt
from torch import nn


def inner_loop(
    net: nn.Module,
    optim: torchopt.Optimizer,
    num_steps: int,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    for _ in range(num_steps):
        support_logit = net(support_x)
        loss = loss_func(support_logit, support_y)
        optim.zero_grad()  # zero gradients
        loss.backward()  # backward
        optim.step()  # step updates
    optim.zero_grad()  # zero gradients


def differentiable_inner_loop(
    net: nn.Module,
    optim: torchopt.MetaOptimizer,
    num_steps: int,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    for _ in range(num_steps):
        support_logit = net(support_x)
        loss = loss_func(support_logit, support_y)
        optim.step(loss)
