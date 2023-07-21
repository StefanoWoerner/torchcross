import random
from collections.abc import Iterable

from torch.utils.data import ChainDataset, IterableDataset


class InterleaveDataset(ChainDataset):

    datasets: Iterable[IterableDataset]

    def __init__(self, datasets: Iterable[IterableDataset]):
        super().__init__(datasets)

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]
        while iterators:
            to_remove = []
            for i in range(len(iterators)):
                try:
                    yield next(iterators[i])
                except StopIteration:
                    to_remove.append(i)
            for i in reversed(to_remove):
                del iterators[i]


class RandomInterleaveDataset(InterleaveDataset):
    def __init__(self, datasets: Iterable[IterableDataset]):
        super().__init__(datasets)

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self.datasets]
        weights = [len(dataset) for dataset in self.datasets]
        while iterators:
            i = random.choices(range(len(iterators)), weights=weights)[0]
            weights[i] -= 1
            try:
                yield next(iterators[i])
            except StopIteration:
                del iterators[i]
                del weights[i]
