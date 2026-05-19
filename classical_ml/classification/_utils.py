"""Shared utilities for the classification subpackage.

Hosts pieces that multiple algorithms reuse:
- numerically-stable `sigmoid` and `softmax`
- `check_random_state` for deterministic RNG
- `OvRClassifier`: wraps a natively-binary estimator into a K-class one
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


# ---------------- Activations ----------------

def sigmoid(z: NDArray) -> NDArray:
    """Numerically-stable logistic sigmoid."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z: NDArray, axis: int = -1) -> NDArray:
    """Numerically-stable softmax along `axis`."""
    z_shift = z - np.max(z, axis=axis, keepdims=True)
    exp_z   = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


# ---------------- RNG helper ----------------

def check_random_state(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a `np.random.Generator`, accepting None / int / Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# ---------------- One-vs-Rest meta-classifier ----------------

class _BinaryEstimator(Protocol):
    """Minimal protocol an OvR base estimator must satisfy."""
    def fit(self, X: NDArray, y: NDArray) -> "_BinaryEstimator": ...
    def decision_function(self, X: NDArray) -> NDArray: ...


class OvRClassifier:
    """One-vs-Rest meta-classifier.

    Given a `base_factory` that produces a fresh binary classifier with a
    `decision_function(X) -> (n,)` method, fit one binary estimator per class
    `c` on the relabeled target `y == c → 1, else 0`. At inference, stack the
    K decision scores into `(n, K)` and pick the argmax.

    The wrapped estimator only ever sees 2-class targets, so there is no
    recursion if the same class is used both as wrapper and base.
    """

    def __init__(self, base_factory: Callable[[], _BinaryEstimator]):
        self.base_factory = base_factory
        self.estimators_: list[_BinaryEstimator] = []
        self.classes_: NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> "OvRClassifier":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for c in self.classes_:
            y_bin = (y == c).astype(int)
            est = self.base_factory()
            est.fit(X, y_bin)
            self.estimators_.append(est)
        return self

    def decision_function(self, X: NDArray) -> NDArray:
        """Return (n, K) matrix of per-class decision scores."""
        assert self.classes_ is not None, "Call fit() before decision_function()."
        return np.column_stack([est.decision_function(X) for est in self.estimators_])

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None, "Call fit() before predict()."
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: NDArray) -> NDArray:
        """Softmax over decision scores. Uncalibrated."""
        return softmax(self.decision_function(X), axis=1)
