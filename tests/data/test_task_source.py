import pytest
import torch
from torch.utils.data import Dataset

from torchcross.data import (
    TaskSource,
    BatchedTaskSource,
    RandomChainTaskSource,
    ConcatTaskSource,
    TaskTarget,
)
from torchcross.data import WrapTaskSource


class DummyDataset(Dataset):
    data = torch.rand(10)
    labels = torch.randint(0, 2, (10,))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


@pytest.fixture
def dummy_dataset():
    return DummyDataset()


class DummyTaskSource(TaskSource):
    data = torch.rand(10)
    labels = torch.randint(0, 2, (10,))
    classes = {0: "a", 1: "b"}
    task_target = TaskTarget.MULTICLASS_CLASSIFICATION
    task_identifier = "test"

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


@pytest.fixture
def dummy_task_source():
    return DummyTaskSource()


class TestWrapTaskSource:
    @pytest.mark.parametrize(
        "task_target",
        [TaskTarget.MULTICLASS_CLASSIFICATION, TaskTarget.BINARY_CLASSIFICATION],
    )
    def test_wrap_task_source_correctly_wraps_given_dataset(
        self, dummy_dataset, task_target
    ):
        classes = {0: "a", 1: "b"}
        wrap_task_source = WrapTaskSource(dummy_dataset, task_target, classes)

        assert wrap_task_source.labels.tolist() == dummy_dataset.labels.tolist()
        assert wrap_task_source.classes == classes
        assert wrap_task_source.task_target == task_target

        for i in range(len(dummy_dataset)):
            assert wrap_task_source[i] == dummy_dataset[i]


class TestBatchedTaskSource:
    @pytest.mark.parametrize("batch_size", [2, 3, 4, 5])
    @pytest.mark.parametrize("shuffle", [True, False])
    def test_batched_task_source_returns_correct_batches(
        self, dummy_dataset, batch_size, shuffle
    ):
        task_source = WrapTaskSource(
            dummy_dataset, TaskTarget.MULTICLASS_CLASSIFICATION, {0: "a", 1: "b"}
        )
        batched_task_source = BatchedTaskSource(task_source, batch_size, shuffle)

        for batch in batched_task_source:
            assert (
                len(batch) == batch_size
                or len(batch) == len(dummy_dataset) % batch_size
            )


class TestRandomChainTaskSource:
    def test_random_chain_task_source_iterates_over_all_batches(self, dummy_dataset):
        task_source = WrapTaskSource(
            dummy_dataset, TaskTarget.MULTICLASS_CLASSIFICATION, {0: "a", 1: "b"}
        )
        batched_task_source1 = BatchedTaskSource(task_source, 2, True)
        batched_task_source2 = BatchedTaskSource(task_source, 3, True)
        random_chain_task_source = RandomChainTaskSource(
            [batched_task_source1, batched_task_source2]
        )

        assert len(random_chain_task_source) == len(batched_task_source1) + len(
            batched_task_source2
        )


class TestConcatTaskSource:
    def test_concat_task_source_returns_correct_item_and_description(
        self, dummy_dataset
    ):
        task_source1 = WrapTaskSource(
            dummy_dataset, TaskTarget.MULTICLASS_CLASSIFICATION, {0: "a", 1: "b"}
        )
        task_source2 = WrapTaskSource(
            dummy_dataset, TaskTarget.BINARY_CLASSIFICATION, {0: "a", 1: "b"}
        )
        concat_task_source = ConcatTaskSource([task_source1, task_source2])

        for i in range(len(concat_task_source)):
            item, description = concat_task_source[i]
            if i < len(task_source1):
                assert item == task_source1[i]
                assert description.task_target == task_source1.task_target
                assert description.classes == task_source1.classes
            else:
                assert item == task_source2[i - len(task_source1)]
                assert description.task_target == task_source2.task_target
                assert description.classes == task_source2.classes
