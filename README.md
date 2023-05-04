# TorchCross
**Easy-to-use PyTorch library for cross-domain learning, few-shot learning and
meta-learning.**

## What is TorchCross?

TorchCross is a PyTorch library for cross-domain learning, few-shot learning and
meta-learning. It provides convenient utilities for creating cross-domain learning
or few-shot learning experiments.

### Package Overview
- `torchcross`: The main package, containing the core functionality of the library.
- `torchcross.data`: Contains the `CrossDomainDataset` and `FewShotDataset`
  classes, which wrap `TaskSource` instances to produce batches for cross-domain
  learning or tasks for few-shot learning experiments.
- `torchcross.data.task`: Contains the `Task` and `TaskDescription` classes, which 
  represent a task in a few-shot learning scenario and a task's metadata, respectively.
- `torchcross.cd` contains functions to create heads, losses and metrics
  for cross-domain learning experiments.

**This library is still in beta. The API is potentially subject to change. Any feedback
is welcome.**

## Installation

The library can be installed via pip:

```bash
pip install torchcross
```


## Examples

See the [`examples`](examples) directory.

