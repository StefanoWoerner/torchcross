import bisect
import random
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
    "RandomChainTaskSource",
    "ConcatTaskSource",
]

from ..utils import to_numpy


class TaskSource(Dataset, ABC):
    """A dataset that can be used as a source to generate tasks."""

    task_target: TaskTarget
    labels: np.ndarray | torch.Tensor | Sequence
    classes: dict[int, str]
    task_identifier: str = ""


class WrapTaskSource(TaskSource):
    """Wraps a dataset as a task source."""

    def __init__(
        self,
        dataset: Dataset,
        task_target: TaskTarget,
        classes: dict[int, str],
        label_fn: Callable[[Dataset], np.ndarray | torch.Tensor | Sequence] = None,
        task_identifier: str = "",
    ):
        self.dataset = dataset
        self.task_target = task_target
        self.classes = classes
        self.task_identifier = task_identifier

        def default_label_fn(ds: Dataset):
            return np.stack([to_numpy(item[1]) for item in ds])

        label_fn = label_fn if label_fn else default_label_fn
        self.labels = label_fn(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class BatchedTaskSource(TaskSource, IterableDataset):
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
        self.task_target = task_source.task_target
        self.classes = task_source.classes
        self.task_identifier = task_source.task_identifier
        self.with_task_description = with_task_description
        if self.with_task_description:
            self.task_description = TaskDescription(
                self.task_target, self.classes, self.task_identifier
            )

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


class RandomChainTaskSource(TaskSource, IterableDataset):
    def __init__(self, task_sources: list[BatchedTaskSource]):
        self.task_sources = task_sources

    def __iter__(self):
        iterators = [iter(ts) for ts in self.task_sources]
        # Calculate the weights of each iterator based on the batched task source length
        weights = [len(ts) for ts in self.task_sources]
        task_descriptions = [
            TaskDescription(ts.task_target, ts.classes, ts.task_identifier)
            for ts in self.task_sources
        ]
        while iterators:
            # Randomly select an iterator with a probability proportional to its weight
            i = random.choices(range(len(self.task_sources)), weights=weights)[0]
            selected_iterator = iterators[i]
            try:
                # Return one element from the selected iterator
                yield next(selected_iterator), task_descriptions[i]
            except StopIteration:
                # The iterator is exhausted, so remove it from the list of iterators
                iterators.remove(selected_iterator)

    def __len__(self):
        return sum(len(ts) for ts in self.task_sources)


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
            TaskDescription(
                self.datasets[dataset_idx].task_target,
                self.datasets[dataset_idx].classes,
                self.datasets[dataset_idx].task_identifier,
            ),
        )

    def __len__(self):
        return sum(len(d) for d in self.datasets)


class SubsetTaskSource(TaskSource):
    def __init__(self, task_source: TaskSource, indices: Sequence[int]):
        self.task_source = task_source
        self.indices = indices
        self.task_target = task_source.task_target
        self.classes = task_source.classes
        self.task_identifier = task_source.task_identifier
        self.labels = task_source.labels[indices]

    def __getitem__(self, idx):
        return self.task_source[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
