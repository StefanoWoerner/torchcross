import bisect
from abc import ABC
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    IterableDataset,
)

from torchcross.utils.collate_fn import identity
from .task import TaskTarget, TaskDescription

__all__ = [
    "TaskSource",
    "WrapTaskSource",
    "BatchedTaskSource",
    "ConcatTaskSource",
]

from ..utils import to_numpy


class TaskSource(Dataset, ABC):
    """A dataset that can be used as a source to generate tasks."""

    task_description: TaskDescription
    labels: np.ndarray | torch.Tensor | Sequence

    def get_num_samples_per_class(
        self, neg=False
    ) -> np.ndarray | torch.Tensor | list[int]:
        task_target = self.task_description.task_target
        total = len(self.labels)

        classes = self.task_description.classes
        num_classes = len(classes)
        max_class_key = max(classes.keys())

        not_implemented_message = f"Task target {task_target} not yet implemented"

        if isinstance(self.labels, np.ndarray):
            if task_target is TaskTarget.MULTICLASS_CLASSIFICATION:
                num_samples = np.bincount(self.labels, minlength=num_classes)
            elif task_target is TaskTarget.MULTILABEL_CLASSIFICATION:
                num_samples = np.sum(self.labels, axis=0)
            elif task_target is TaskTarget.BINARY_CLASSIFICATION:
                num_samples = np.sum(self.labels, axis=0)
            elif task_target is TaskTarget.ORDINAL_REGRESSION:
                num_samples = np.bincount(self.labels, minlength=max_class_key + 1)
                num_samples = num_samples[np.array(list(classes.keys()), dtype=np.int_)]
            else:
                raise NotImplementedError(not_implemented_message)
            return num_samples - total if neg else num_samples

        elif isinstance(self.labels, torch.Tensor):
            if task_target is TaskTarget.MULTICLASS_CLASSIFICATION:
                num_samples = torch.bincount(self.labels, minlength=num_classes)
            elif task_target is TaskTarget.MULTILABEL_CLASSIFICATION:
                num_samples = torch.sum(self.labels, dim=0)
            elif task_target is TaskTarget.BINARY_CLASSIFICATION:
                num_samples = torch.sum(self.labels, dim=0)
            elif task_target is TaskTarget.ORDINAL_REGRESSION:
                num_samples = torch.bincount(self.labels, minlength=max_class_key + 1)
                num_samples = num_samples[
                    torch.tensor(
                        list(classes.keys()),
                        device=num_samples.device,
                        dtype=torch.long,
                    )
                ]
            else:
                raise NotImplementedError(not_implemented_message)
            return num_samples - total if neg else num_samples

        elif isinstance(self.labels, Sequence):
            if task_target is TaskTarget.MULTICLASS_CLASSIFICATION:
                num_samples = [sum(l == c for l in self.labels) for c in classes]
            elif task_target is TaskTarget.MULTILABEL_CLASSIFICATION:
                num_samples = [sum(l[c] for l in self.labels) for c in classes]
            elif task_target is TaskTarget.BINARY_CLASSIFICATION:
                num_samples = [
                    sum(l[0] if isinstance(l, Sequence) else l for l in self.labels)
                ]
            elif task_target is TaskTarget.ORDINAL_REGRESSION:
                num_samples = [sum(l == c for l in self.labels) for c in classes]
            else:
                raise NotImplementedError(not_implemented_message)
            return [n - total for n in num_samples] if neg else num_samples

        else:
            raise ValueError("Unsupported label data type")


class WrapTaskSource(TaskSource):
    """Wraps a dataset as a task source."""

    def __init__(
        self,
        dataset: Dataset,
        task_target: TaskTarget,
        classes: dict[int, str],
        label_fn: Callable[[Dataset], np.ndarray | torch.Tensor | Sequence] = None,
        task_identifier: str = "",
        domain_identifier: str = "",
    ):
        self.dataset = dataset
        self.task_description = TaskDescription(
            task_target, classes, task_identifier, domain_identifier
        )

        def default_label_fn(ds: Dataset):
            return np.stack([to_numpy(item[1]) for item in ds])

        label_fn = label_fn if label_fn else default_label_fn
        self.labels = label_fn(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class BatchedTaskSource(IterableDataset):
    def __init__(
        self,
        task_source: TaskSource,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        collate_fn=None,
        with_task_description: bool = False,
    ):
        self.task_source = task_source
        self.task_description = task_source.task_description
        self.with_task_description = with_task_description

        sampler = (
            RandomSampler(self.task_source)
            if shuffle
            else SequentialSampler(self.task_source)
        )
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.collate_fn = collate_fn if collate_fn else identity

    def __iter__(self):
        for batch in self.batch_sampler:
            if self.with_task_description:
                yield (
                    self.collate_fn([self.task_source[i] for i in batch]),
                    self.task_description,
                )
            else:
                yield self.collate_fn([self.task_source[i] for i in batch])

    def __len__(self):
        return len(self.batch_sampler)


class ConcatTaskSource(TaskSource, ConcatDataset):
    datasets: list[TaskSource]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return (
            self.datasets[dataset_idx][sample_idx],
            self.datasets[dataset_idx].task_description,
        )

    def __len__(self):
        return sum(len(d) for d in self.datasets)


class SubsetTaskSource(TaskSource):
    def __init__(self, task_source: TaskSource, indices: Sequence[int]):
        self.task_source = task_source
        self.indices = indices
        self.task_description = task_source.task_description
        self.labels = task_source.labels[indices]

    def __getitem__(self, idx):
        return self.task_source[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
