import random
from abc import ABC
import bisect
from collections.abc import Sequence

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


class TaskSource(Dataset, ABC):
    task_target: TaskTarget
    labels: np.ndarray | torch.Tensor | Sequence
    classes: dict[int, str] = None
    task_identifier: str = ""


class BatchedTaskSource(TaskSource, IterableDataset):
    def __init__(
        self,
        task_source: TaskSource,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        collate_fn=None,
    ):
        self.task_source = task_source
        self.task_target = task_source.task_target
        self.classes = task_source.classes
        self.task_identifier = task_source.task_identifier

        sampler = (
            RandomSampler(self.task_source)
            if shuffle
            else SequentialSampler(self.task_source)
        )
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.collate_fn = collate_fn if collate_fn else identity

    def __iter__(self):
        for batch in self.batch_sampler:
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
            TaskDescription(
                ts.task_target, ts.classes, ts.task_identifier
            ) for ts in self.task_sources
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
