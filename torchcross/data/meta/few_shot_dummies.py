import torch

from torchcross.utils.collate_fn import identity
from . import MetaDataset
from ..base import TaskSource
from ..task import TaskTarget, Task


class TakeFirstFewShotMetaDataset(MetaDataset):
    def __init__(
        self,
        dataset: TaskSource,
        collate_fn=None,
        n_shot_per_class: int = 5,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.task_target = dataset.task_target

        if self.task_target is TaskTarget.MULTICLASS_CLASSIFICATION:
            self.indices = {
                c: [i for i, l in enumerate(self.dataset.labels) if l == c]
                for c in self.dataset.classes
            }
        else:
            raise NotImplementedError(
                f"Task target {self.task_target} not yet implemented"
            )

        def multiclass_n_shot(start=0):
            sample = [
                (i, j)
                for j, c in enumerate(self.dataset.classes)
                for i in self.indices[c][start : start + n_shot_per_class]
            ]
            return [(self.dataset[i][0], torch.tensor(j)) for i, j in sample]

        self.collate_fn = collate_fn if collate_fn else identity

        self.support = self.collate_fn(multiclass_n_shot(0))
        self.query = self.collate_fn(multiclass_n_shot(n_shot_per_class))

    def __getitem__(self, _) -> Task:
        return Task(self.support, self.query, self.task_target, self.dataset.classes)
