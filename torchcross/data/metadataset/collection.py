from typing import overload, Iterable

from torch.utils.data import Dataset

from torchcross.data import TaskDescription, Task
from torchcross.data.metadataset import MetaDataset


class CollectionMetaDataset(MetaDataset):
    tasks: list[Task]

    @overload
    def __init__(self, tasks: Iterable[Task]) -> None:
        ...

    @overload
    def __init__(self, *tasks: Task) -> None:
        ...

    @overload
    def __init__(
        self,
        support_datasets: Iterable[Dataset],
        query_datasets: Iterable[Dataset],
        task_descriptions: Iterable[TaskDescription],
    ) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if len(args) == 1 and not kwargs:
            (tasks,) = args
            assert len(tasks) > 0
            self.tasks = list(tasks)
        elif len(args) == 3 and not kwargs:
            support_datasets, query_datasets, task_descriptions = args
            assert (
                len(support_datasets) == len(query_datasets) == len(task_descriptions)
            )
            assert len(support_datasets) > 0
            self.tasks = [
                Task(
                    support,
                    query,
                    task_description.task_target,
                    task_description.classes,
                )
                for support, query, task_description in zip(
                    support_datasets, query_datasets, task_descriptions
                )
            ]

        else:
            raise TypeError(f"Expected 1 or 3 positional arguments, got {len(args)}")

    def __getitem__(self, index: int) -> Task:
        return self.tasks[index]
