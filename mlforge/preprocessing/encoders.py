"""Categorical encoders (work on strings as well as numbers)."""
import numpy as np


class LabelEncoder:
    """Encode target labels as integers ``0 .. n_classes-1``."""

    _validate_io = False

    def fit(self, y):
        self.classes_ = np.unique(np.asarray(y))
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        try:
            return np.array([self._map[v] for v in np.asarray(y)])
        except KeyError as exc:
            raise ValueError(f"y contains previously unseen label: {exc}") from None

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded):
        return self.classes_[np.asarray(y_encoded, dtype=int)]


def _as_2d(X):
    X = np.asarray(X)
    return X.reshape(-1, 1) if X.ndim == 1 else X


class OrdinalEncoder:
    """Encode each categorical column as integers."""

    _validate_io = False

    def fit(self, X):
        X = _as_2d(X)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = _as_2d(X)
        out = np.zeros(X.shape, dtype=float)
        for j, cats in enumerate(self.categories_):
            m = {c: i for i, c in enumerate(cats)}
            try:
                out[:, j] = [m[v] for v in X[:, j]]
            except KeyError as exc:
                raise ValueError(f"Unknown category {exc} in column {j}") from None
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=int)
        return np.column_stack([self.categories_[j][X[:, j]] for j in range(X.shape[1])])


class OneHotEncoder:
    """One-hot encode one or several categorical columns.

    ``handle_unknown='ignore'`` encodes unseen categories as all zeros.
    """

    _validate_io = False

    def __init__(self, handle_unknown="error", drop_first=False):
        self.handle_unknown = handle_unknown
        self.drop_first = drop_first

    def fit(self, X):
        X = _as_2d(X)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        self.classes_ = self.categories_[0]       # v0.1 compatibility (1-D usage)
        return self

    def transform(self, X):
        X = _as_2d(X)
        blocks = []
        for j, cats in enumerate(self.categories_):
            m = {c: i for i, c in enumerate(cats)}
            block = np.zeros((X.shape[0], len(cats)))
            for r, v in enumerate(X[:, j]):
                if v in m:
                    block[r, m[v]] = 1
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category {v!r} in column {j}")
            blocks.append(block[:, 1:] if self.drop_first else block)
        return np.hstack(blocks)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z)
        cols, pos = [], 0
        for cats in self.categories_:
            width = len(cats) - (1 if self.drop_first else 0)
            block = Z[:, pos:pos + width]
            if self.drop_first:
                block = np.hstack([1 - block.sum(1, keepdims=True), block])
            cols.append(cats[np.argmax(block, axis=1)])
            pos += width
        return np.column_stack(cols)

    def get_feature_names(self, input_names=None):
        names = []
        for j, cats in enumerate(self.categories_):
            base = input_names[j] if input_names else f"x{j}"
            names += [f"{base}={c}" for c in (cats[1:] if self.drop_first else cats)]
        return names
