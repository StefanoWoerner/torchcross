import pytest
import torch
from torchmetrics.classification import (
    MulticlassAUROC,
    MultilabelAUROC,
    BinaryAUROC,
    MulticlassAccuracy,
    MultilabelAccuracy,
    BinaryAccuracy,
)

from torchcross.data.task import TaskTarget, TaskDescription
from torchcross.cd.metrics import (
    IndexedMulticlassAccuracy,
    Accuracy,
    IndexedMulticlassAUROC,
    AUROC,
)

# Sample task descritptions for testing
classes_multiclass = {0: "A", 1: "B", 2: "C"}
classes_binary = {0: "Negative", 1: "Positive"}
classes_multilabel = {0: "X", 1: "Y", 2: "Z"}
classes_ordinal = {1: "Low", 5: "Medium", 8: "High"}
multiclass_description = TaskDescription(
    TaskTarget.MULTICLASS_CLASSIFICATION,
    classes_multiclass,
    "multiclass_task",
    "multiclass_domain",
)
binary_description = TaskDescription(
    TaskTarget.BINARY_CLASSIFICATION,
    classes_binary,
    "binary_task",
    "binary_domain",
)
multilabel_description = TaskDescription(
    TaskTarget.MULTILABEL_CLASSIFICATION,
    classes_multilabel,
    "multilabel_task",
    "multilabel_domain",
)
ordinal_description = TaskDescription(
    TaskTarget.ORDINAL_REGRESSION,
    classes_ordinal,
    "ordinal_task",
    "ordinal_domain",
)


# Mock tensors for testing
multiclass_preds = torch.randn(10, len(classes_multiclass))
multiclass_targets = torch.randint(0, len(classes_multiclass), (10,))

binary_preds = torch.randn(10, len(classes_binary))
binary_targets = torch.randint(0, len(classes_binary), (10,))

multilabel_preds = torch.randn(10, len(classes_multilabel))
multilabel_targets = torch.randint(0, len(classes_multilabel), (10,))

ordinal_preds = torch.randn(10, len(classes_ordinal))
ordinal_target_indices = torch.randint(0, len(classes_ordinal), (10,))
index_to_ordinal_class = {i: c for i, c in enumerate(classes_ordinal)}
ordinal_targets = torch.tensor(
    [index_to_ordinal_class[t.item()] for t in ordinal_target_indices]
)


class TestIndexedMulticlassAccuracy:
    @pytest.fixture
    def metric(self):
        # Create a metric instance
        return IndexedMulticlassAccuracy(classes_ordinal)

    @pytest.fixture
    def base_metric(self):
        # Create a base metric instance
        return MulticlassAccuracy(len(classes_ordinal), average="micro")

    def test_init(self):
        # Test that the metric initializes correctly
        metric = IndexedMulticlassAccuracy(classes_multiclass)
        assert metric.num_classes == len(classes_multiclass)

    def test_forward(self, metric, base_metric):
        # Test that the metric computes the correct value
        m_result = metric(ordinal_preds, ordinal_targets)
        bm_result = base_metric(ordinal_preds, ordinal_target_indices)
        assert m_result == bm_result
        assert metric.compute() == base_metric.compute()


class TestAccuracy:
    @pytest.mark.parametrize(
        "task_description,expected",
        [
            (multiclass_description, MulticlassAccuracy),
            (binary_description, BinaryAccuracy),
            (multilabel_description, MultilabelAccuracy),
            (ordinal_description, IndexedMulticlassAccuracy),
        ],
    )
    def test_init(self, task_description, expected):
        # Test that the metric initializes correctly
        metric = Accuracy(task_description)
        assert isinstance(metric, expected)
        if expected == MulticlassAccuracy or expected == IndexedMulticlassAccuracy:
            assert metric.num_classes == len(task_description.classes)
        elif expected == MultilabelAccuracy:
            assert metric.num_labels == len(task_description.classes)
        elif expected == BinaryAccuracy:
            with pytest.raises(AttributeError):
                metric.num_classes
        else:
            pytest.fail(f"Unexpected metric type: {expected}")


class TestIndexedMulticlassAUROC:
    @pytest.fixture
    def metric(self):
        # Create a metric instance
        return IndexedMulticlassAUROC(classes_ordinal)

    @pytest.fixture
    def base_metric(self):
        # Create a base metric instance
        return MulticlassAUROC(len(classes_ordinal))

    def test_init(self):
        # Test that the metric initializes correctly
        metric = IndexedMulticlassAUROC(classes_multiclass)
        assert metric.num_classes == len(classes_multiclass)

    def test_forward(self, metric, base_metric):
        # Test that the metric computes the correct value
        m_result = metric(ordinal_preds, ordinal_targets)
        bm_result = base_metric(ordinal_preds, ordinal_target_indices)
        assert m_result == bm_result
        assert metric.compute() == base_metric.compute()


class TestAUROC:
    @pytest.mark.parametrize(
        "task_description,expected",
        [
            (multiclass_description, MulticlassAUROC),
            (binary_description, BinaryAUROC),
            (multilabel_description, MultilabelAUROC),
            (ordinal_description, IndexedMulticlassAUROC),
        ],
    )
    def test_init(self, task_description, expected):
        # Test that the metric initializes correctly
        metric = AUROC(task_description)
        assert isinstance(metric, expected)
        if expected == MulticlassAUROC or expected == IndexedMulticlassAUROC:
            assert metric.num_classes == len(task_description.classes)
        elif expected == MultilabelAUROC:
            assert metric.num_labels == len(task_description.classes)
        elif expected == BinaryAUROC:
            with pytest.raises(AttributeError):
                metric.num_classes
        else:
            pytest.fail(f"Unexpected metric type: {expected}")
