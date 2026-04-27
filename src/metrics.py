"""Error metrics for benchmark summaries."""

import numpy as np


def mae(errors):
    """Return the mean absolute error."""
    values = np.asarray(errors, dtype=float)
    return float(np.mean(np.abs(values)))


def rmse(errors):
    """Return the root mean squared error."""
    values = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def max_absolute_error(errors):
    """Return the largest absolute error."""
    values = np.asarray(errors, dtype=float)
    return float(np.max(np.abs(values)))

