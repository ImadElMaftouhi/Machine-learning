"""
K-nearest neighbors classifier.
Lazy learner: no computation at fit time. All work happens at predict time,
making it O(n_train * d) per query — fast to train, slow to predict on large data.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class KNN:
    def __init__(self, k: int = 3, distance: str = "euclidean",
                 weights: str = "uniform"):
        """
        Parameters
        ----------
        k         : number of neighbors to consider
        distance  : 'euclidean' | 'manhattan' | 'cosine'
        weights   : 'uniform'  — majority vote
                    'distance' — closer neighbors vote with weight 1/d
        """
        if not (isinstance(k, int) and k > 0):
            raise ValueError(f"k must be a positive integer. Got {k!r}.")
        if distance not in ("euclidean", "manhattan", "cosine"):
            raise ValueError(f"distance must be 'euclidean', 'manhattan', or 'cosine'. Got {distance!r}.")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance'. Got {weights!r}.")

        self.k        = k
        self.distance = distance
        self.weights  = weights

        self.X_train:      NDArray | None = None
        self.y_train:      NDArray | None = None
        self.classes_:     NDArray | None = None
        self._cls_to_idx:  dict    | None = None   # label → column index

    # guard ────────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.X_train is None:
            raise RuntimeError("Call fit() before predict.")

    # fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: NDArray, y: NDArray) -> "KNN":
        self.X_train   = np.asarray(X, dtype=float)
        self.y_train   = np.asarray(y)
        self.classes_  = np.unique(self.y_train)
        self._cls_to_idx = {label: i for i, label in enumerate(self.classes_)}
        return self

    # distance matrix ──────────────────────────────────────────────────────

    def _distance_matrix(self, X_te: NDArray, X_tr: NDArray) -> NDArray:
        """
        Compute the full (n_te, n_tr) distance matrix without any Python loop
        over samples.

        Euclidean uses the identity:
            ||a - b||² = ||a||² + ||b||² - 2 a·b
        which reduces to a single matrix multiply.

        Manhattan and cosine require O(n_te * n_tr * d) intermediate arrays —
        acceptable for moderate sizes; for very large data switch to chunked
        evaluation or scipy.spatial.distance.cdist.
        """
        if self.distance == "euclidean":
            sq_te = (X_te ** 2).sum(axis=1, keepdims=True)    # (n_te, 1)
            sq_tr = (X_tr ** 2).sum(axis=1)                   # (n_tr,)
            D2    = sq_te + sq_tr - 2.0 * (X_te @ X_tr.T)    # (n_te, n_tr)
            return np.sqrt(np.clip(D2, 0.0, None))            # clip guards float rounding

        elif self.distance == "manhattan":
            # (n_te, n_tr, d) — sum along last axis
            return np.abs(X_te[:, None, :] - X_tr[None, :, :]).sum(axis=2)

        else:  # cosine
            X_te_n = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-12)
            X_tr_n = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-12)
            return 1.0 - X_te_n @ X_tr_n.T                   # similarity → distance

    # predict_proba ────────────────────────────────────────────────────────

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Returns
        -------
        binary      → (n,)    P(y = classes_[1])
        multi-class → (n, K)  one column per class, rows sum to 1
        """
        self._check_fitted()
        assert self.X_train    is not None
        assert self.y_train    is not None
        assert self.classes_   is not None
        assert self._cls_to_idx is not None

        X   = np.atleast_2d(np.asarray(X, dtype=float))
        D   = self._distance_matrix(X, self.X_train)   # (n_te, n_tr)
        K   = len(self.classes_)
        out = np.zeros((len(X), K), dtype=float)

        for i, dists in enumerate(D):
            k_idx    = np.argsort(dists)[: self.k]
            k_dists  = dists[k_idx]
            k_labels = self.y_train[k_idx]

            if self.weights == "distance":
                # exact-zero distance → treat as infinitely close
                w = np.where(k_dists == 0.0, 1e12, 1.0 / k_dists)
            else:
                w = np.ones(self.k)

            for label, weight in zip(k_labels, w):
                out[i, self._cls_to_idx[label]] += weight

            out[i] /= out[i].sum()

        return out[:, 1] if K == 2 else out   # sklearn convention for binary

    # predict ──────────────────────────────────────────────────────────────

    def predict(self, X: NDArray) -> NDArray:
        self._check_fitted()
        assert self.classes_ is not None

        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            # proba is (n,)  P(class=1)
            return self.classes_[(proba >= 0.5).astype(int)]
        else:
            return self.classes_[np.argmax(proba, axis=1)]

    # score ────────────────────────────────────────────────────────────────

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # repr ─────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return f"KNN(k={self.k}, distance={self.distance!r}, weights={self.weights!r})"

    def __repr__(self) -> str:
        return self.__str__()
