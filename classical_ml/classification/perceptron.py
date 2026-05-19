"""Rosenblatt's perceptron — the original online linear classifier (1958).

Native binary classifier; for K>2 classes it dispatches to a One-vs-Rest
meta-classifier internally.

Algorithm (binary):
    For each epoch, iterate samples in order. Map labels to {-1, +1}.
    Predict y_hat = sign(w·x + b). On misclassification (y · (w·x + b) <= 0):
        w += lr * y * x
        b += lr * y
    Convergence is *guaranteed in finite steps* iff the data is linearly
    separable. On non-separable data, the loop never converges — `n_iter`
    is a hard cap.

`predict_proba` returns sigmoid/softmax of the decision score — uncalibrated,
exposed only for API compatibility with the rest of the package.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

try:
    from ._utils import OvRClassifier, check_random_state, sigmoid, softmax
except ImportError:
    # Allows `python perceptron.py` from inside the classification dir,
    # matching the bare-import convention used by evaluate_performance.py.
    from _utils import OvRClassifier, check_random_state, sigmoid, softmax  # type: ignore[no-redef]


class Perceptron:
    def __init__(
        self,
        lr: float = 1.0,
        n_iter: int = 100,
        shuffle: bool = True,
        random_state: int | None = 42,
        init: Literal["zeros", "random"] | str = "zeros",
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.init = init

        # Set in fit()
        self.classes_: NDArray | None = None
        self.weights_: NDArray | None = None          # binary only
        self.bias_: float = 0.0                       # binary only
        self.errors_per_epoch_: list[int] = []        # binary only
        self.n_features_: int | None = None
        self._ovr: OvRClassifier | None = None        # set in multi-class case

    def __repr__(self) -> str:
        return (f"Perceptron(lr={self.lr}, n_iter={self.n_iter}, "
                f"shuffle={self.shuffle}, init={self.init!r})")

    # ---------------- Public API ----------------

    def fit(self, X: NDArray, y: NDArray) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        K = len(self.classes_)

        if K < 2:
            raise ValueError("Perceptron requires at least 2 classes.")

        if K == 2:
            self._fit_binary(X, y)
        else:
            # OvR: each sub-perceptron is itself a Perceptron in binary mode.
            # The sub-fits see 2 unique labels, so no recursion.
            self._ovr = OvRClassifier(base_factory=self._make_binary_clone)
            self._ovr.fit(X, y)
        return self

    def decision_function(self, X: NDArray) -> NDArray:
        """Raw decision scores.

        Binary  : shape (n,) — positive class is `classes_[1]`.
        Multi   : shape (n, K) — column k corresponds to `classes_[k]`.
        """
        X = np.asarray(X, dtype=float)
        if self._ovr is not None:
            return self._ovr.decision_function(X)
        self._check_binary_fitted()
        assert self.weights_ is not None
        return X @ self.weights_ + self.bias_

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None, "Call fit() before predict()."
        if self._ovr is not None:
            return self._ovr.predict(X)
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]

    def predict_proba(self, X: NDArray) -> NDArray:
        """Uncalibrated pseudo-probabilities.

        Binary  : sigmoid(score)        → shape (n,)
        Multi   : softmax(scores)       → shape (n, K)
        """
        if self._ovr is not None:
            return self._ovr.predict_proba(X)
        return sigmoid(self.decision_function(X))

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ---------------- Internals ----------------

    def _check_binary_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("Call fit() before using this method.")

    def _make_binary_clone(self) -> "Perceptron":
        """Return a fresh Perceptron with identical hyperparameters."""
        return Perceptron(
            lr=self.lr, n_iter=self.n_iter, shuffle=self.shuffle,
            random_state=self.random_state, init=self.init,
        )

    def _fit_binary(self, X: NDArray, y: NDArray) -> None:
        assert self.classes_ is not None
        n, d = X.shape

        # Map labels → {-1, +1}: classes_[0] → -1, classes_[1] → +1.
        y_signed = np.where(y == self.classes_[1], 1, -1).astype(float)

        rng = check_random_state(self.random_state)
        if self.init == "zeros":
            self.weights_ = np.zeros(d)
        else:
            self.weights_ = rng.normal(scale=0.01, size=d)
        self.bias_ = 0.0
        self.errors_per_epoch_ = []

        indices = np.arange(n)

        assert self.weights_ is not None
        for _ in range(self.n_iter):
            if self.shuffle:
                rng.shuffle(indices)

            mistakes = 0
            for i in indices:
                xi, yi = X[i], y_signed[i]
                # Misclassified ⟺ yi · (w·xi + b) ≤ 0
                if yi * (xi @ self.weights_ + self.bias_) <= 0.0:
                    self.weights_ += self.lr * yi * xi
                    self.bias_   += self.lr * yi #type:ignore
                    mistakes += 1

            self.errors_per_epoch_.append(mistakes)
            if mistakes == 0:
                break  # data is linearly separable & converged


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("Smoke test — Perceptron\n" + "-" * 32)
    for n_cls, label in [(2, "binary"), (4, "multinomial")]:
        X, y = make_classification(
            n_samples=800, n_features=8, n_informative=6, n_redundant=1,
            n_classes=n_cls, n_clusters_per_class=1, class_sep=1.2,
            random_state=42,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

        clf = Perceptron(lr=1.0, n_iter=50).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        # Cross-check against sklearn for the same setup.
        from sklearn.linear_model import Perceptron as SkPerceptron
        sk = SkPerceptron(max_iter=50, eta0=1.0, random_state=42).fit(X_tr, y_tr)
        sk_acc = accuracy_score(y_te, sk.predict(X_te))
        print(f"{label:>11}: ours={acc:.4f}   sklearn={sk_acc:.4f}")


if __name__ == "__main__":
    main()
