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
    support: Dataset | Collection
    query: Dataset | Collection
    task_target: TaskTarget
    classes: dict[int, str] = None


@dataclass
class TaskDescription:
    task_target: TaskTarget
    classes: dict[int, str] = None
    task_identifier: str = ""
