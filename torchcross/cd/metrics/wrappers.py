from typing import Any

from torch import nn
from torchmetrics import Metric
from torchmetrics.wrappers.abstract import WrapperMetric

from torchcross.data import TaskDescription


class CrossDomainWrapper(WrapperMetric):
    """Wraps around a metric constructor and applies it to each task in a cross-domain setting.

    Args:
        metric_constructor: The class of the metric to wrap around.
        task_descriptions: The task descriptions for each task.
        metric_args: The arguments to pass to the metric constructor.
        metric_kwargs: The keyword arguments to pass to the metric constructor.
    """

    is_differentiable = False

    def __init__(
        self,
        metric_constructor: type[Metric],
        task_descriptions: list[TaskDescription] | None = None,
        metric_args: tuple[Any] = (),
        metric_kwargs: dict[str, Any] | None = None,
        auto_add_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.metric_constructor = metric_constructor
        self.task_descriptions = task_descriptions
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs or {}
        self.auto_add_metrics = auto_add_metrics

        self.metrics = nn.ModuleDict()
        if task_descriptions is not None:
            for task_description in task_descriptions:
                self.add_metric(task_description)

    def add_metric(self, task_description: TaskDescription) -> None:
        """Add a metric for a new task.

        Args:
            task_description: The task description for the new task.
        """
        task_identifier = task_description.task_identifier
        if task_identifier in self.metrics:
            raise ValueError(f"Task {task_identifier} already has a metric.")
        self.metrics[task_identifier] = self.metric_constructor(
            task_description,
            *self.metric_args,
            **self.metric_kwargs,
        ).to(self.device)

    def update(self, task_description: TaskDescription, *args, **kwargs) -> None:
        """Update the metric for a task.

        Args:
            task_description: The task description for the task.
            args: The positional arguments to pass to the metric.
            kwargs: The keyword arguments to pass to the metric.
        """
        task_identifier = task_description.task_identifier
        if task_identifier not in self.metrics:
            if self.auto_add_metrics:
                self.add_metric(task_description)
            else:
                raise ValueError(f"Task {task_identifier} does not have a metric.")
        self.metrics[task_identifier].update(*args, **kwargs)

    def compute(self) -> Any:
        """Compute the metric for each task.

        Returns:
            The computed metric for each task.
        """
        return {
            task_identifier: metric.compute()
            for task_identifier, metric in self.metrics.items()
        }

    def forward(self, task_description: TaskDescription, *args, **kwargs) -> Any:
        """Forward input to the metric for a task.

        Args:
            task_description: The task description for the task.
            args: The positional arguments to pass to the metric.
            kwargs: The keyword arguments to pass to the metric.

        Returns:
            The output of the metric for the task.
        """
        task_identifier = task_description.task_identifier
        if task_identifier not in self.metrics:
            if self.auto_add_metrics:
                self.add_metric(task_description)
            else:
                raise ValueError(f"Task {task_identifier} does not have a metric.")
        self._forward_cache = self.metrics[task_identifier](*args, **kwargs)
        return self._forward_cache

    def reset(self) -> None:
        """Reset all underlying metrics."""
        for metric in self.metrics.values():
            metric.reset()
        super().reset()


class CrossDomainMeanWrapper(CrossDomainWrapper):
    """Wraps around a metric constructor and applies it to each task in a cross-domain setting.
    Computes the mean of the metric across tasks.
    """

    def compute(self) -> Any:
        """Compute the mean of the metric over all tasks.

        Returns:
            The mean of the computed metric over all tasks.
        """
        return sum(m.compute() for m in self.metrics.values()) / len(self.metrics)
