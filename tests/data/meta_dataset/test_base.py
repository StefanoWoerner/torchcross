import pytest
import torch
from torch.utils.data import Dataset, TensorDataset
from torchcross.data.metadataset import (
    MetaDataset,
    MetaConcatDataset,
    IterableMetaDataset,
    MetaChainDataset,
)
from torchcross.data import TaskSource, Task, TaskTarget


class DummyTaskSource(TaskSource):
    data = torch.rand(10)
    labels = torch.randint(0, 2, (10,))
    classes = {0: "a", 1: "b"}
    task_target = TaskTarget.MULTICLASS_CLASSIFICATION
    task_identifier = "test"

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class DummyMetaDataset(MetaDataset):
    task_source = DummyTaskSource()

    def __getitem__(self, index):
        return Task(
            self.task_source[index],
            self.task_source[-index],
            self.task_source.task_target,
            self.task_source.classes,
        )


class DummyIterableMetaDataset(IterableMetaDataset):
    task_source = DummyTaskSource()

    def __iter__(self):
        for i in range(10):
            yield Task(
                self.task_source[i],
                self.task_source[-i],
                self.task_source.task_target,
                self.task_source.classes,
            )


class TestMetaDataset:
    def test_get_item(self):
        ds = DummyMetaDataset()
        task = ds[0]
        assert isinstance(task, Task)

    def test_addition(self):
        ds1 = DummyMetaDataset()
        ds2 = DummyMetaDataset()
        concat_ds = ds1 + ds2
        assert isinstance(concat_ds, MetaConcatDataset)


class TestMetaConcatDataset:
    def test_init(self):
        with pytest.raises(AssertionError):
            MetaConcatDataset([])

        ds1 = DummyMetaDataset()
        ds2 = DummyMetaDataset()
        concat_ds = MetaConcatDataset([ds1, ds2])
        assert len(concat_ds.datasets) == 2

    def test_get_item(self):
        ds1 = DummyMetaDataset()
        ds2 = DummyMetaDataset()
        concat_ds = MetaConcatDataset([ds1, ds2])
        task = concat_ds[(0, 0)]
        assert isinstance(task, Task)


class TestIterableMetaDataset:
    def test_iter(self):
        ds = DummyIterableMetaDataset()
        tasks = list(ds)
        assert len(tasks) == 10
        assert all(isinstance(task, Task) for task in tasks)

    def test_addition(self):
        ds1 = DummyIterableMetaDataset()
        ds2 = DummyIterableMetaDataset()
        chain_ds = ds1 + ds2
        assert isinstance(chain_ds, MetaChainDataset)


class TestMetaChainDataset:
    def test_init(self):
        ds1 = DummyIterableMetaDataset()
        ds2 = DummyIterableMetaDataset()
        chain_ds = MetaChainDataset([ds1, ds2])
        assert len(chain_ds.datasets) == 2
