from collections.abc import Sequence

import numpy as np
import torch


def to_numpy(arg, /):
    """Convert torch tensors, lists, tuples, other Sequences and sets to
    numpy arrays. Numpy arrays are returned as is. Single numbers are
    converted to numpy arrays of size 1. Collections are converted
    recursively.
    """

    if isinstance(arg, np.ndarray):
        return arg  # Already a numpy array, no conversion necessary
    elif isinstance(arg, torch.Tensor):
        return arg.detach().cpu().numpy()  # Make sure to detach and move to CPU
    elif isinstance(arg, (list, tuple, Sequence)):
        return np.array([to_numpy(item) for item in arg])
    elif isinstance(arg, set):
        return np.array([to_numpy(item) for item in sorted(arg)])
    # think about how to convert dict and other mappings
    # elif isinstance(collection, dict):
    #     return {key: to_numpy(value) for key, value in collection.items()}
    else:
        # This will work for single numbers, etc., but may fail with other types
        return np.array(arg)
