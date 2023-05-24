import math
from collections import OrderedDict
from collections.abc import Sequence, Callable, Iterable
from typing import Optional, Literal

import torch
import torchopt
from torch import nn, Tensor

from torchcross.cd.heads import new_classification_head
from torchcross.cd.losses import get_loss_func
from torchcross.data.task import TaskDescription, Task
from torchcross.models.cross_domain import CrossDomainModel
from torchcross.models.episodic import EpisodicModel
from torchcross.utils.layers import Expand

head_init_t = Literal[
    "default",
    "kaiming",
    "zero",
    "adaptive",
    "adaptive normal",
    "adaptive uniform",
    "unicorn",
    "proto",
]


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


class MAML(EpisodicModel):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        task_description: TaskDescription,
        inner_optimizer: Callable[[torch.nn.Module], torchopt.MetaOptimizer],
        eval_inner_optimizer: Callable[
            [Iterable[torch.nn.Parameter]], torchopt.Optimizer
        ],
        num_inner_steps: int,
        eval_num_inner_steps: Optional[int] = None,
        transductive: bool = True,
        reset_head: Optional[head_init_t] = None,
    ) -> None:
        super(MAML, self).__init__()

        self.backbone = backbone
        self.num_backbone_features = num_backbone_features
        self.head = new_classification_head(
            task_description.task_target,
            task_description.classes,
            num_backbone_features,
            init="kaiming",
        )
        self.net = torch.nn.Sequential(
            OrderedDict(backbone=self.backbone, head=self.head)
        )
        self.inner_loss_func = get_loss_func(
            task_description.task_target, task_description.classes
        )

        self.inner_optimizer: torchopt.MetaOptimizer = inner_optimizer(self.net)
        self.eval_inner_optimizer: torchopt.Optimizer = eval_inner_optimizer(
            self.net.parameters()
        )
        self.num_inner_steps = num_inner_steps
        self.eval_num_inner_steps = eval_num_inner_steps or num_inner_steps

        self.transductive = transductive
        self.reset_head_strategy = reset_head

    def forward(self, batch: Sequence[Task]) -> list[torch.Tensor]:
        net_state_dict = torchopt.extract_state_dict(
            self.net, by="reference", detach_buffers=True
        )

        query_logits = []

        for task in batch:
            ql = self.episode(task)
            query_logits.append(ql)
            torchopt.recover_state_dict(self.net, net_state_dict)

        return query_logits

    def episode(
        self, task: Task, with_support_output=False
    ) -> Tensor | tuple[Tensor, Tensor]:
        support_x, support_y = task.support
        query_x = task.query[0]

        if self.reset_head_strategy is not None:
            self.reset_head()
        net = self.net
        net.train()
        if self.training:
            inner_optim_state_dict = torchopt.extract_state_dict(
                self.inner_optimizer, by="reference"
            )
            differentiable_inner_loop(
                net,
                self.inner_optimizer,
                self.num_inner_steps,
                support_x,
                support_y,
                self.inner_loss_func,
            )
            torchopt.recover_state_dict(self.inner_optimizer, inner_optim_state_dict)
        else:
            inner_loop(
                net,
                self.eval_inner_optimizer,
                self.eval_num_inner_steps,
                support_x,
                support_y,
                self.inner_loss_func,
            )
        if not self.transductive:
            net.eval()
        query_logits = net(query_x)
        if with_support_output:
            with torch.no_grad():
                support_logits = net(support_x)
            return support_logits, query_logits
        return query_logits

    def reset_head(self):
        if (
            self.reset_head_strategy == "default"
            or self.reset_head_strategy == "kaiming"
        ):
            self.head.reset_parameters()
        elif self.reset_head_strategy == "zero":
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)
        elif (
            self.reset_head_strategy == "adaptive"
            or self.reset_head_strategy == "adaptive normal"
        ):
            torch.nn.init.normal_(
                self.head.weight,
                self.head.weight.mean().item(),
                self.head.weight.std().item(),
            )
            torch.nn.init.normal_(
                self.head.bias,
                self.head.bias.mean().item(),
                self.head.bias.std().item(),
            )
        elif self.reset_head_strategy == "adaptive uniform":
            weigth_mean = self.head.weight.mean().item()
            weight_std = self.head.weight.std().item()
            wa = weigth_mean - math.sqrt(3) * weight_std
            wb = weigth_mean + math.sqrt(3) * weight_std
            torch.nn.init.uniform_(self.head.weight, wa, wb)
            bias_mean = self.head.bias.mean().item()
            bias_std = self.head.bias.std().item()
            ba = bias_mean - math.sqrt(3) * bias_std
            bb = bias_mean + math.sqrt(3) * bias_std
            torch.nn.init.uniform_(self.head.bias, ba, bb)
        elif self.reset_head_strategy == "unicorn":
            raise NotImplementedError
        elif self.reset_head_strategy == "proto":
            raise NotImplementedError
        else:
            raise ValueError(f"Head reset option {self.reset_head_strategy} unknown.")


class CrossDomainMAML(EpisodicModel, CrossDomainModel):
    def __init__(
        self,
        backbone: nn.Module,
        num_backbone_features: int,
        inner_optimizer,
        eval_inner_optimizer,
        num_inner_steps: int,
        eval_num_inner_steps: Optional[int] = None,
        transductive: bool = True,
        head_init: Optional[head_init_t] = None,
    ) -> None:
        super(CrossDomainMAML, self).__init__()

        self.backbone = backbone
        self.num_backbone_features = num_backbone_features

        self.inner_optim_constructor = inner_optimizer
        self.eval_inner_optim_constructor = eval_inner_optimizer
        self.num_inner_steps = num_inner_steps
        self.eval_num_inner_steps = eval_num_inner_steps or num_inner_steps

        self.transductive = transductive
        self.head_init = head_init or "zero"
        self.input_module = Expand(-1, 3, -1, -1)

    def get_net(self, task_description: TaskDescription) -> torch.nn.Module:
        return torch.nn.Sequential(
            OrderedDict(
                input_module=self.input_module,
                backbone=self.backbone,
                head=new_classification_head(
                    task_description.task_target,
                    task_description.classes,
                    self.num_backbone_features,
                    device=self.device_dummy.device,
                    init=self.head_init,
                ),
            )
        )

    def forward(self, batch: Sequence[Task]) -> list[torch.Tensor]:
        backbone_state_dict = torchopt.extract_state_dict(
            self.backbone, by="reference", detach_buffers=True
        )

        query_logits = []

        for task in batch:
            query_logit = self.episode(task)
            query_logits.append(query_logit)
            torchopt.recover_state_dict(self.backbone, backbone_state_dict)

        return query_logits

    def episode(
        self, task: Task, with_support_output=False
    ) -> Tensor | tuple[Tensor, Tensor]:
        support_x, support_y = task.support
        query_x = task.query[0]

        net = self.get_net(TaskDescription(task.task_target, task.classes))
        inner_loss_func = get_loss_func(
            task.task_target, task.classes, self.device_dummy.device
        )
        net.train()
        if self.training:
            differentiable_inner_loop(
                net,
                self.inner_optim_constructor(net),
                self.num_inner_steps,
                support_x,
                support_y,
                inner_loss_func,
            )
        else:
            inner_loop(
                net,
                self.eval_inner_optim_constructor(net.parameters()),
                self.eval_num_inner_steps,
                support_x,
                support_y,
                inner_loss_func,
            )
        if not self.transductive:
            net.eval()
        query_logits = net(query_x)
        if with_support_output:
            with torch.no_grad():
                support_logits = net(support_x)
            return support_logits, query_logits
        return query_logits
