import itertools
import random
from builtins import NotImplementedError
from collections.abc import Sequence, Iterable, Iterator, Callable
from functools import partial
from typing import TypeVar

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from torchcross.utils.collate_fn import identity
from . import MetaDataset
from .base import IterableMetaDataset
from ..task_source import TaskSource
from ..task import TaskTarget, Task


LT = TypeVar("LT", bound=torch.Tensor | np.ndarray | Sequence[int])


def get_indices(
    labels: LT, classes, task_target: TaskTarget
) -> tuple[dict[int, LT], dict[int, LT]]:
    if isinstance(labels, np.ndarray):

        def ind_func(x):
            return np.nonzero(x)[0]

    elif isinstance(labels, torch.Tensor):

        def ind_func(x):
            return torch.nonzero(x).flatten()

    else:
        raise ValueError("Unsupported label data type")

    if task_target is TaskTarget.MULTICLASS_CLASSIFICATION:
        pos_indices = {c: ind_func(labels == c) for c in classes}
        neg_indices = {c: ind_func(labels != c) for c in classes}
    elif task_target is TaskTarget.MULTILABEL_CLASSIFICATION:
        pos_indices = {c: ind_func(labels[:, c]) for c in classes}
        neg_indices = {c: ind_func(labels[:, c] - 1) for c in classes}
    elif task_target is TaskTarget.BINARY_CLASSIFICATION:
        pos_indices = {1: ind_func(labels)}
        neg_indices = {1: ind_func(labels - 1)}
    elif task_target is TaskTarget.ORDINAL_REGRESSION:
        pos_indices = {c: ind_func(labels == c) for c in classes}
        neg_indices = {c: ind_func(labels != c) for c in classes}
    else:
        raise NotImplementedError(f"Task target {task_target} not yet implemented")
    return pos_indices, neg_indices


def multiclass_few_shot_sample(
    pos_indices: dict[int, np.ndarray | torch.Tensor | Sequence[int]],
    n_shot: int,
    selected_classes: Sequence[int] | dict[int, str],
):
    sample = np.array([], dtype=np.int64)
    # Sample n_shot positive examples for each class
    for c in selected_classes:
        pos_sample = np.random.choice(pos_indices[c], size=n_shot, replace=False)
        sample = np.concatenate([sample, pos_sample])
    return sample


def binary_few_shot_sample(
    pos_indices: dict[int, np.ndarray | torch.Tensor | Sequence[int]],
    neg_indices: dict[int, np.ndarray | torch.Tensor | Sequence[int]],
    n_shot: int,
):
    pos_sample = np.random.choice(pos_indices[1], size=n_shot, replace=False)
    neg_sample = np.random.choice(neg_indices[1], size=n_shot, replace=False)
    return np.concatenate([pos_sample, neg_sample])


def multilabel_few_shot_sample(
    labels: np.ndarray | torch.Tensor,
    pos_indices: dict[int, np.ndarray | torch.Tensor | Sequence[int]],
    neg_indices: dict[int, np.ndarray | torch.Tensor | Sequence[int]],
    n_shot: int,
    selected_classes: Sequence[int] | dict[int, str],
) -> np.ndarray:
    if isinstance(labels, np.ndarray):
        sum_func = partial(np.sum, axis=0)
        all_func = np.all
    elif isinstance(labels, torch.Tensor):
        sum_func = partial(torch.sum, dim=0)
        all_func = torch.all
    else:
        raise ValueError("Unsupported data type")

    if isinstance(selected_classes, dict):
        selected_classes = list(selected_classes.keys())
    labels = labels[:, selected_classes]

    def criterion(sample_to_test: np.ndarray) -> bool:
        column_sums = sum_func(labels[sample_to_test])
        pos_criterion = all_func(column_sums >= n_shot)
        neg_criterion = all_func(column_sums <= len(sample_to_test) - n_shot)
        return pos_criterion and neg_criterion

    sample = np.array([], dtype=np.int64)

    # Sample n_shot positive and n_shot negative examples for each label
    for c in selected_classes:
        pos_sample = np.random.choice(pos_indices[c], size=n_shot, replace=False)
        neg_sample = np.random.choice(neg_indices[c], size=n_shot, replace=False)
        sample = np.concatenate([sample, pos_sample, neg_sample])

    # Remove duplicates and shuffle the sample for the later steps
    sample = np.unique(sample)
    np.random.shuffle(sample)

    # Add single examples from the sample until criterion is satisfied, i.e. until
    # there are enough positive and negative examples for each label.
    # We do this so we have to remove fewer examples (and therefore need to compute the
    # criterion fewer times) later on.
    for i in range(2 * n_shot, len(sample)):
        small_sample = sample[:i]
        if criterion(small_sample):
            sample = small_sample
            break

    # Remove examples (greedily) as long as criterion is satisfied without the example,
    # i.e. as long as there are enough positive and negative examples left for each
    # label.
    i = 0
    while i < len(sample):
        row_removed = np.delete(sample, i)
        if criterion(row_removed):
            sample = row_removed
        else:
            i += 1

    return sample


def remove_indices(
    pos_indices: dict[int, np.ndarray | torch.Tensor | Sequence[int]],
    neg_indices: dict[int, np.ndarray | torch.Tensor | Sequence[int]],
    sample: np.ndarray | torch.Tensor | Sequence[int],
):
    sample_set = set(sample)

    def _compute(class_indices):
        if isinstance(class_indices, np.ndarray):
            return np.setdiff1d(class_indices, sample)
        elif isinstance(class_indices, torch.Tensor):
            return torch.from_numpy(np.setdiff1d(class_indices, sample))
            # return class_indices[
            #     torch.isin(class_indices, sample, assume_unique=True, invert=True)
            # ]
        else:
            return [i for i in class_indices if i not in sample_set]

    new_pos_indices = {c: _compute(c_indices) for c, c_indices in pos_indices.items()}
    new_neg_indices = {c: _compute(c_indices) for c, c_indices in neg_indices.items()}
    return new_pos_indices, new_neg_indices


class FewShotMetaDataset(IterableMetaDataset):
    def __init__(
        self,
        task_source: TaskSource,
        collate_fn=None,
        n_support_samples_per_class: int = 5,
        n_query_samples_per_class: int = 20,
        filter_classes_min_samples: int = 50,
        length: int | None = None,
        output_indices: bool = False,
    ) -> None:
        super().__init__()
        self.task_source = task_source
        self.task_target = task_source.task_target

        self.pos_indices, self.neg_indices = get_indices(
            self.task_source.labels, self.task_source.classes, self.task_target
        )

        self.used_classes = list(self.task_source.classes.keys())
        if self.task_target is TaskTarget.BINARY_CLASSIFICATION:
            self.used_classes = [1]
        if filter_classes_min_samples > 0:
            self.used_classes = [
                c
                for c in self.used_classes
                if len(self.pos_indices[c]) >= filter_classes_min_samples
            ]

            if self.task_target in (
                TaskTarget.BINARY_CLASSIFICATION,
                TaskTarget.MULTILABEL_CLASSIFICATION,
            ):
                self.used_classes = [
                    c
                    for c in self.used_classes
                    if len(self.neg_indices[c]) >= filter_classes_min_samples
                ]
        if (
            len(self.used_classes) < 2
            and self.task_target is not TaskTarget.BINARY_CLASSIFICATION
        ):
            message = "FewShotMetaDataset requires at least 2 classes"
            if filter_classes_min_samples > 0:
                message += f" with at least {filter_classes_min_samples} samples each"
            message += f"for {self.task_target}."
            raise ValueError(message)

        self.collate_fn = collate_fn if collate_fn else identity
        self.n_support_samples_per_class = n_support_samples_per_class
        self.n_query_samples_per_class = n_query_samples_per_class
        self.length = length
        self.output_indices = output_indices

    def __len__(self) -> int:
        if self.length is None:
            raise ValueError("Length is not set.")
        return self.length

    def __iter__(self) -> Iterator[Task]:
        match self.task_target:
            case TaskTarget.MULTICLASS_CLASSIFICATION | TaskTarget.ORDINAL_REGRESSION:
                return self._multiclass_iter()
            case TaskTarget.MULTILABEL_CLASSIFICATION:
                return self._multilabel_iter()
            case TaskTarget.BINARY_CLASSIFICATION:
                return self._binary_iter()
            case target:
                msg = f"Task target {target} is not supported by FewShotMetaDataset."
                raise NotImplementedError(msg)

    def _multiclass_iter(self) -> Iterator[Task]:
        pos_indices = self.pos_indices
        neg_indices = self.neg_indices
        iter_index = 0
        stop = False
        while not stop:
            try:
                support_indices = multiclass_few_shot_sample(
                    pos_indices, self.n_support_samples_per_class, self.used_classes
                )
                new_pos_indices, new_neg_indices = remove_indices(
                    pos_indices, neg_indices, support_indices
                )
                query_indices = multiclass_few_shot_sample(
                    new_pos_indices, self.n_query_samples_per_class, self.used_classes
                )
                if self.length is None:
                    pos_indices, neg_indices = remove_indices(
                        new_pos_indices, new_neg_indices, query_indices
                    )
                else:
                    iter_index += 1
                    if iter_index >= self.length:
                        stop = True
                class_to_label = {
                    c: torch.tensor(ci) for ci, c in enumerate(self.used_classes)
                }
                support = [
                    (
                        i if self.output_indices else self.task_source[i][0],
                        class_to_label[self.task_source.labels[i].item()],
                    )
                    for i in support_indices
                ]
                query = [
                    (
                        i if self.output_indices else self.task_source[i][0],
                        class_to_label[self.task_source.labels[i].item()],
                    )
                    for i in query_indices
                ]
                support = self.collate_fn(support)
                query = self.collate_fn(query)
                class_dict = {
                    ci: self.task_source.classes[c]
                    for ci, c in enumerate(self.used_classes)
                }
                yield Task(support, query, self.task_target, class_dict)
            except ValueError:
                return

    def _multilabel_iter(self) -> Iterator[Task]:
        pos_indices = self.pos_indices
        neg_indices = self.neg_indices
        iter_index = 0
        stop = False
        while not stop:
            try:
                support_indices = multilabel_few_shot_sample(
                    self.task_source.labels,
                    pos_indices,
                    neg_indices,
                    self.n_support_samples_per_class,
                    self.used_classes,
                )
                new_pos_indices, new_neg_indices = remove_indices(
                    pos_indices, neg_indices, support_indices
                )
                query_indices = multilabel_few_shot_sample(
                    self.task_source.labels,
                    new_pos_indices,
                    new_neg_indices,
                    self.n_query_samples_per_class,
                    self.used_classes,
                )
                if self.length is None:
                    pos_indices, neg_indices = remove_indices(
                        new_pos_indices, new_neg_indices, query_indices
                    )
                else:
                    iter_index += 1
                    if iter_index >= self.length:
                        stop = True

                labels = self.task_source.labels[:, self.used_classes]
                support = [
                    (
                        i if self.output_indices else self.task_source[i][0],
                        torch.tensor(labels[i]),
                    )
                    for i in support_indices
                ]
                query = [
                    (
                        i if self.output_indices else self.task_source[i][0],
                        torch.tensor(labels[i]),
                    )
                    for i in query_indices
                ]
                support = self.collate_fn(support)
                query = self.collate_fn(query)
                class_dict = {
                    ci: self.task_source.classes[c]
                    for ci, c in enumerate(self.used_classes)
                }
                yield Task(support, query, self.task_target, class_dict)
            except ValueError:
                return

    def _binary_iter(self) -> Iterator[Task]:
        pos_indices = self.pos_indices
        neg_indices = self.neg_indices
        iter_index = 0
        stop = False
        while not stop:
            try:
                support_indices = binary_few_shot_sample(
                    pos_indices, neg_indices, self.n_support_samples_per_class
                )
                new_pos_indices, new_neg_indices = remove_indices(
                    pos_indices, neg_indices, support_indices
                )
                query_indices = binary_few_shot_sample(
                    new_pos_indices, new_neg_indices, self.n_query_samples_per_class
                )
                if self.length is None:
                    pos_indices, neg_indices = remove_indices(
                        new_pos_indices, new_neg_indices, query_indices
                    )
                else:
                    iter_index += 1
                    if iter_index >= self.length:
                        stop = True

                labels = self.task_source.labels
                support = [
                    (
                        i if self.output_indices else self.task_source[i][0],
                        torch.tensor(labels[i]),
                    )
                    for i in support_indices
                ]
                query = [
                    (
                        i if self.output_indices else self.task_source[i][0],
                        torch.tensor(labels[i]),
                    )
                    for i in query_indices
                ]
                support = self.collate_fn(support)
                query = self.collate_fn(query)
                class_dict = {
                    0: self.task_source.classes[0],
                    1: self.task_source.classes[1],
                }
                yield Task(support, query, self.task_target, class_dict)

            except ValueError:
                return


class SubTaskRandomFewShotMetaDataset(FewShotMetaDataset):
    def __init__(
        self,
        task_source: TaskSource,
        collate_fn=None,
        *,
        n_support_samples_per_class_min: int = 1,
        n_support_samples_per_class_max: int = 10,
        n_query_samples_per_class: int = 10,
        filter_classes_min_samples: int = 40,
        length: int | None = None,
        output_indices: bool = False,
    ) -> None:
        super().__init__(
            task_source,
            collate_fn,
            n_support_samples_per_class_min,
            n_query_samples_per_class,
            filter_classes_min_samples,
            length,
            output_indices,
        )
        self.original_used_classes = self.used_classes
        self.n_support_samples_per_class_min = n_support_samples_per_class_min
        self.n_support_samples_per_class_max = n_support_samples_per_class_max

    def __iter__(self):
        if self.task_target != TaskTarget.BINARY_CLASSIFICATION:
            n_way = random.randint(2, len(self.original_used_classes))
            self.used_classes = random.sample(self.original_used_classes, n_way)
        self.n_support_samples_per_class = random.randint(
            self.n_support_samples_per_class_min, self.n_support_samples_per_class_max
        )
        for task in super().__iter__():
            yield task
            if self.task_target != TaskTarget.BINARY_CLASSIFICATION:
                n_way = random.randint(2, len(self.original_used_classes))
                self.used_classes = random.sample(self.original_used_classes, n_way)
            self.n_support_samples_per_class = random.randint(
                self.n_support_samples_per_class_min,
                self.n_support_samples_per_class_max,
            )
        self.used_classes = self.original_used_classes
