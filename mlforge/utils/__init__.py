from .gradcheck import check_gradient, numerical_gradient, relative_error
from .math_utils import (
    correlation_matrix,
    cosine_similarity,
    covariance_matrix,
    dot_product,
    euclidean_distance,
    manhattan_distance,
    mean,
    sigmoid,
    softmax,
    std,
    variance,
)
from .pairwise import pairwise_distances
from .validation import NotFittedError, check_array, check_X_y

__all__ = [
    "mean", "variance", "std", "euclidean_distance", "manhattan_distance",
    "dot_product", "cosine_similarity", "covariance_matrix", "correlation_matrix",
    "softmax", "sigmoid", "check_array", "check_X_y", "NotFittedError",
    "pairwise_distances", "numerical_gradient", "check_gradient", "relative_error",
]
