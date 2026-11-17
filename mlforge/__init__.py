"""
MLForge: Machine Learning From Scratch
======================================

Pure-NumPy implementations of classical machine-learning algorithms, built
to be read, verified (numerical gradient checks, scikit-learn parity tests)
and benchmarked.
"""

__version__ = "0.3.0"

from . import (
    autograd,
    base,
    bayes,
    cluster,
    datasets,
    decomposition,
    linear_model,
    metrics,
    model_selection,
    neighbors,
    neural_network,
    optimization,
    pipeline,
    preprocessing,
    svm,
    tree,
    utils,
)
from .base import BaseEstimator
from .pipeline import Pipeline
from .utils.validation import NotFittedError

# `visualization` is imported lazily on first access so that matplotlib is optional.


def __getattr__(name):
    if name == "visualization":
        import importlib
        return importlib.import_module(".visualization", __name__)
    raise AttributeError(f"module 'mlforge' has no attribute {name!r}")


__all__ = [
    "base", "utils", "datasets", "autograd", "optimization", "metrics", "preprocessing",
    "model_selection", "linear_model", "tree", "neighbors", "cluster", "decomposition",
    "bayes", "svm", "neural_network", "pipeline", "visualization", "Pipeline",
    "BaseEstimator", "NotFittedError", "__version__",
]
