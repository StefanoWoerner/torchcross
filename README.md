# TorchCross
**Easy-to-use PyTorch library for cross-domain learning, few-shot learning and
meta-learning.**

## What is TorchCross?

TorchCross is a PyTorch library for cross-domain learning, few-shot learning and
meta-learning. It provides convenient utilities for creating cross-domain learning
or few-shot learning experiments.

### Package Overview
- `torchcross`: The main package, containing the library.
- `torchcross.cd`: Contains functions to create network heads, losses and metrics
  for cross-domain learning experiments.
- `torchcross.data`: Contains classes to load data as for cross-domain learning
  or few-shot learning experiments.
  - `torchcross.data.task`: Contains the `Task` and `TaskDescription` classes, which 
    represent a task in a few-shot learning scenario and a task's metadata,
    respectively.
  - `torchcross.data.task_source`: Contains the `TaskSource` class, which extends the 
    `torch.utils.data.Dataset` class and represents a data source for sampling tasks
    in a few-shot learning scenario. Additionally, it contains the utility classes
    which facilitate working with `TaskSource` objects.
  - `torchcross.data.metadataset`: Contains the `MetaDataset` class, which
    represents a collection of tasks in a few-shot learning scenario. The
    `FewShotMetaDataset` and `SubTaskRandomFewShotMetaDataset` classes extend the
    `MetaDataset` class and provide a convenient interface for sampling few-shot
    instances from a `TaskSource`.
- `torchcross.models`: Contains `torch.nn.Module` classes for cross-domain
  and few-shot scenarios.
- `torchcross.models.lightning`: Contains `pytorch_lightning.LightningModule` classes
  which wrap the `torch.nn.Module` classes in `torchcross.models` and provide
  convenient training and evaluation routines.
- `torchcross.utils`: Contains various utility functions.

**This library is still in beta. The API is potentially subject to change. Any feedback
is welcome.**

## Installation

The library can be installed via pip:

```bash
pip install torchcross
```


## Basic Usage Examples

### Wrapping a PyTorch Dataset as a TaskSource

The `WrapTaskSource` class can be used to wrap a PyTorch dataset as a `TaskSource`.

```python
import torch.utils.data
from torchcross.data import TaskTarget, WrapTaskSource

# Create a PyTorch dataset
dataset: torch.utils.data.Dataset = ...

# Define the appropriate TaskTarget. Let's assume that the task is a multi-class
# classification task.
task_target = TaskTarget.MULTICLASS_CLASSIFICATION

# Classes can be provided as a dictionary. Let's assume that the dataset contains
# four classes with names "Class A", "Class B", "Class C" and "Class D".
classes = {0: "Class A", 1: "Class B", 2: "Class C", 3: "Class D"}

# Wrap the dataset as a TaskSource
task_source = WrapTaskSource(dataset, task_target, classes)
```

### Creating a Few-Shot Meta-Dataset

The `FewShotMetaDataset` class can be used to create a few-shot meta-dataset from a
`TaskSource`.

```python
from torchcross.data.metadataset import FewShotMetaDataset
import torchcross as tx


# Create a TaskSource.
task_source = ...

# Create a few-shot meta-dataset from the task source. Let's assume that we want to
# sample 100 tasks, each containing 5 support samples and 10 query samples.
# Let's use the stack function from the tx.utils module to stack the support and query
# samples into a single tensor.
meta_dataset = FewShotMetaDataset(
    task_source,
    collate_fn=tx.utils.collate_fn.stack,
    n_support_samples_per_class=5,
    n_query_samples_per_class=10,
    filter_classes_min_samples=30,
    length=100,
)
```

### Creating a Few-Shot Meta-Dataset with Random Sub-Tasks

The `SubTaskRandomFewShotMetaDataset` class can be used to create a few-shot
meta-dataset from a `TaskSource` where each task is a random sub-task of the original
task, meaning that each few-shot task contains a random subset of the original task's
classes.

```python
from torchcross.data.metadataset import SubTaskRandomFewShotMetaDataset
import torchcross as tx


# Create a TaskSource.
task_source = ...

# Create a few-shot meta-dataset from the task source. Let's assume that we want to
# sample 100 tasks, each containing a random subset of the original task's classes and
# 1 to 10 support samples per class and 10 query samples per class.
# Let's use the stack function from the tx.utils module to stack the support and query
# samples into a single tensor.
few_shot = SubTaskRandomFewShotMetaDataset(
    task_source,
    collate_fn=tx.utils.collate_fn.stack,
    n_support_samples_per_class_min=1,
    n_support_samples_per_class_max=10,
    n_query_samples_per_class=10,
    filter_classes_min_samples=30,
    length=100,
)
```


### Use the MAML Algorithm to Train a Model on a Few-Shot Meta-Dataset

The `lightning.LightningModule` subclasses in `torchcross.models.lightning` can be used
to train a model on a meta-dataset. Let's assume that we want to train a model on a
few-shot meta-dataset using the MAML algorithm. We can use the `MAML` class from the
`torchcross.models.lightning` module to do so.

```python
from functools import partial

import lightning.pytorch as pl
import torch
import torchopt
from torch.utils.data import DataLoader

from torchcross.data import TaskDescription, TaskTarget
from torchcross.models.lightning import MAML
from torchcross.utils.collate_fn import identity

# Create few-shot meta-datasets for training and validation.
train_meta_dataset = ...
val_meta_dataset = ...

# Create dataloaders for training and validation.
train_dataloader = DataLoader(train_meta_dataset, batch_size=4, collate_fn=identity)
val_dataloader = DataLoader(val_meta_dataset, batch_size=4, collate_fn=identity)

# Create a `TaskDescription` object which describes the task type of task that we want
# to solve. Let's assume that the task is a multi-class classification task with four
# classes.
task_description = TaskDescription(
    TaskTarget.MULTICLASS_CLASSIFICATION,
    classes={0: "Class A", 1: "Class B", 2: "Class C", 3: "Class D"},
)

# Create a backbone network. We could for example use a pre-trained resnet18 backbone.
backbone = ...
num_backbone_output_features = 512

# Create inner and outer optimizers and hyperparameters for MAML.
outer_optimizer = partial(torch.optim.Adam, lr=0.001)
inner_optimizer = partial(torchopt.MetaSGD, lr=0.1)
eval_inner_optimizer = partial(torch.optim.SGD, lr=0.1)
num_inner_steps = 4
eval_num_inner_steps = 32

# Create the lighting model with pre-trained resnet18 backbone
model = MAML(
    (backbone, num_backbone_output_features),
    task_description,
    outer_optimizer,
    inner_optimizer,
    eval_inner_optimizer,
    num_inner_steps,
    eval_num_inner_steps,
)

# Create the lightning trainer
trainer = pl.Trainer(
    inference_mode=False,
    max_epochs=10,
    check_val_every_n_epoch=1,
    val_check_interval=1000,
    limit_val_batches=100,
)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)
```


## Real-World Example

See the [MIMeta](https://www.github.com/StefanoWoerner/mimeta-pytorch) library for a
real-world example of how to use TorchCross to perform cross-domain learning, few-shot
learning or meta-learning experiments.
