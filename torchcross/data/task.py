from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum, auto

from torch.utils.data import Dataset


class TaskTarget(Enum):
    MULTICLASS_CLASSIFICATION = auto()
    BINARY_CLASSIFICATION = auto()
    MULTILABEL_CLASSIFICATION = auto()
    ORDINAL_REGRESSION = auto()
    REGRESSION = auto()
    BINARY_SEGMENTATION = auto()
    MULTILABEL_SEGMENTATION = auto()


@dataclass
class TaskDescription:
    """A task description with a task target and classes as well as
    optional string identifiers for the task and domain.

    Args:
        task_target: The task target.
        classes: The classes for the task.
        task_identifier: The task identifier.
        domain_identifier: The domain identifier.
    """

    task_target: TaskTarget
    classes: dict[int, str] = None
    task_identifier: str = None
    domain_identifier: str = None


@dataclass
class Task:
    """A task with support and query datasets as well as a task
    description.

    Args:
        support: The support dataset or collection of inputs and targets
        query: The query dataset or collection of inputs and targets
        description: The task description
    """

    support: Dataset | Collection
    query: Dataset | Collection
    description: TaskDescription
