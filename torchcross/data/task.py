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
class Task:
    """A task with support and query datasets as well as a task target
    and classes.

    Args:
        support: The support dataset or collection of inputs and targets.
        query: The query dataset or collection of inputs and targets.
        task_target: The task target.
        classes: The classes for the task.
    """

    support: Dataset | Collection
    query: Dataset | Collection
    task_target: TaskTarget
    classes: dict[int, str] = None


@dataclass
class TaskDescription:
    """A task description with a task target and classes as well as an
    optional string identifier for the task.

    Args:
        task_target: The task target.
        classes: The classes for the task.
        task_identifier: The task identifier.
    """

    task_target: TaskTarget
    classes: dict[int, str] = None
    task_identifier: str = ""
