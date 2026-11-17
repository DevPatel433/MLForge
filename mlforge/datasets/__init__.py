from .mnist import load_mnist
from .synthetic import (
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
    make_spirals,
)

__all__ = ["load_mnist", "make_blobs", "make_moons", "make_circles", "make_spirals",
           "make_regression", "make_classification"]
