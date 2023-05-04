from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from torch.utils.data import Dataset, IterableDataset, ChainDataset

from ..base import TaskSource
from ..task import Task


class MetaDataset(Dataset[Task], ABC):
    task_source: TaskSource

    @abstractmethod
    def __getitem__(self, index) -> Task:
        ...

    def __add__(self, other: "MetaDataset") -> "MetaConcatDataset":
        return MetaConcatDataset([self, other])


class MetaConcatDataset(MetaDataset):
    datasets: list[MetaDataset]

    def __init__(self, datasets: Iterable[MetaDataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0

    def __getitem__(self, index: tuple[int, Any]) -> Task:
        outer_index, inner_index = index
        return self.datasets[outer_index][inner_index]


class IterableMetaDataset(IterableDataset[Task], ABC):
    task_source: TaskSource

    @abstractmethod
    def __iter__(self) -> Iterator[Task]:
        ...

    def __add__(self, other: "IterableMetaDataset") -> "MetaChainDataset":
        return MetaChainDataset([self, other])


class MetaChainDataset(ChainDataset, IterableMetaDataset, ABC):
    pass
