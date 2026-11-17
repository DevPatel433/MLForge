"""Feature engineering transformers."""
from itertools import combinations, combinations_with_replacement

import numpy as np

from ..base import BaseEstimator


class PolynomialFeatures(BaseEstimator):
    """Generate polynomial and interaction features up to ``degree``."""

    _allow_1d = False

    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def fit(self, X, y=None):
        d = X.shape[1]
        comb = combinations if self.interaction_only else combinations_with_replacement
        start = 0 if self.include_bias else 1
        self.powers_ = [c for deg in range(start, self.degree + 1) for c in comb(range(d), deg)]
        self.n_output_features_ = len(self.powers_)
        return self

    def transform(self, X):
        cols = [np.prod(X[:, list(c)], axis=1) if c else np.ones(X.shape[0]) for c in self.powers_]
        return np.column_stack(cols)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
