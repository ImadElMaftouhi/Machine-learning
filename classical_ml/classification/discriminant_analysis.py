"""Linear and Quadratic Discriminant Analysis — Gaussian generative classifiers.

Both model the class-conditional density p(x | y=k) as a multivariate Gaussian
and apply Bayes' rule:
    P(y=k | x) ∝ π_k · N(x; μ_k, Σ_k)

LDA assumes a *shared* covariance Σ across classes → linear decision boundary.
QDA allows a per-class Σ_k → quadratic decision boundary.

Fit is closed-form MLE; no iterations. The posterior is the exact Gaussian
posterior so probabilities are naturally calibrated — contrast with Naive
Bayes' independence assumption, which produces over-confident outputs.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

try:
    from ._utils import softmax
except ImportError:
    from _utils import softmax  # type: ignore[no-redef]


class _DiscriminantBase:
    """Shared fit/predict skeleton for LDA and QDA.

    Subclasses implement `_fit_post` (compute covariance-related attrs) and
    `_discriminant` (return (n, K) discriminant scores).
    """

    def __init__(
        self,
        reg: float = 1e-6,
        priors: Union[Literal["empirical", "uniform"], NDArray] = "empirical",
    ):
        self.reg = reg
        self.priors = priors

        # Set in fit()
        self.classes_: NDArray | None = None
        self.n_features_: int | None = None
        self.log_priors_: NDArray | None = None
        self.means_: NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> "_DiscriminantBase":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n, d = X.shape
        self.n_features_ = d

        # --- priors ---
        if isinstance(self.priors, str):
            if self.priors == "empirical":
                _, counts = np.unique(y, return_counts=True)
                self.log_priors_ = np.log(counts / n)
            elif self.priors == "uniform":
                self.log_priors_ = np.full(K, -np.log(K))
            else:
                raise ValueError(
                    f"priors must be 'empirical', 'uniform', or array. "
                    f"Got {self.priors!r}."
                )
        else:
            p = np.asarray(self.priors, dtype=float)
            if p.shape != (K,) or not np.isclose(p.sum(), 1.0):
                raise ValueError(
                    f"Custom priors must have shape ({K},) and sum to 1."
                )
            self.log_priors_ = np.log(p + 1e-12)

        # --- per-class means ---
        self.means_ = np.array(
            [X[y == c].mean(axis=0) for c in self.classes_]
        )  # shape (K, d)

        self._fit_post(X, y)
        return self

    # Subclass hooks
    def _fit_post(self, X: NDArray, y: NDArray) -> None:
        raise NotImplementedError

    def _discriminant(self, X: NDArray) -> NDArray:
        """Return discriminant scores δ_k(x) — shape (n, K)."""
        raise NotImplementedError

    # --- public predict API ---

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None, "Call fit() before predict()."
        scores = self._discriminant(np.asarray(X, dtype=float))
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: NDArray) -> NDArray:
        """Native Gaussian posterior via softmax over discriminants.

        Binary  → shape (n,)  P(y = classes_[1])
        Multi   → shape (n, K)
        """
        assert self.classes_ is not None, "Call fit() before predict_proba()."
        scores = self._discriminant(np.asarray(X, dtype=float))
        proba = softmax(scores, axis=1)
        return proba[:, 1] if len(self.classes_) == 2 else proba

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class LDA(_DiscriminantBase):
    """Linear Discriminant Analysis — shared covariance.

    Decision rule: δ_k(x) = x^T Σ⁻¹ μ_k − ½ μ_k^T Σ⁻¹ μ_k + log π_k
    Linear in x; precomputed as `coef_ @ x + intercept_`.
    """

    def __init__(
        self, reg: float = 1e-6,
        priors: Union[Literal["empirical", "uniform"], NDArray] = "empirical",
    ):
        super().__init__(reg=reg, priors=priors)
        self.cov_: NDArray | None = None
        self.coef_: NDArray | None = None       # (K, d) = Σ⁻¹ μ_k stacked
        self.intercept_: NDArray | None = None  # (K,)

    def __repr__(self) -> str:
        return f"LDA(reg={self.reg!r}, priors={self.priors!r})"

    def _fit_post(self, X: NDArray, y: NDArray) -> None:
        assert self.classes_ is not None and self.means_ is not None
        assert self.log_priors_ is not None
        K = len(self.classes_)
        n, d = X.shape

        # Pooled within-class covariance (unbiased: divide by n - K).
        cov = np.zeros((d, d))
        for i, c in enumerate(self.classes_):
            diff = X[y == c] - self.means_[i]
            cov += diff.T @ diff
        cov /= max(n - K, 1)
        cov += self.reg * np.eye(d)
        self.cov_ = cov

        cov_inv = np.linalg.inv(cov)
        self.coef_ = self.means_ @ cov_inv                                  # (K, d)
        quad      = np.einsum("kd,kd->k", self.coef_, self.means_)          # μ_k^T Σ⁻¹ μ_k
        self.intercept_ = -0.5 * quad + self.log_priors_                    # (K,)

    def _discriminant(self, X: NDArray) -> NDArray:
        assert self.coef_ is not None and self.intercept_ is not None
        return X @ self.coef_.T + self.intercept_                           # (n, K)


class QDA(_DiscriminantBase):
    """Quadratic Discriminant Analysis — per-class covariance.

    Decision rule:
        δ_k(x) = -½ log|Σ_k| - ½ (x - μ_k)^T Σ_k⁻¹ (x - μ_k) + log π_k
    """

    def __init__(
        self, reg: float = 1e-6,
        priors: Union[Literal["empirical", "uniform"], NDArray] = "empirical",
    ):
        super().__init__(reg=reg, priors=priors)
        self.covs_: NDArray | None = None      # (K, d, d)
        self.cov_inv_: NDArray | None = None   # (K, d, d)
        self.log_det_: NDArray | None = None   # (K,)

    def __repr__(self) -> str:
        return f"QDA(reg={self.reg!r}, priors={self.priors!r})"

    def _fit_post(self, X: NDArray, y: NDArray) -> None:
        assert self.classes_ is not None and self.means_ is not None
        K = len(self.classes_)
        _, d = X.shape

        covs = np.zeros((K, d, d))
        for i, c in enumerate(self.classes_):
            Xc   = X[y == c]
            diff = Xc - self.means_[i]
            n_c  = len(Xc)
            covs[i] = diff.T @ diff / max(n_c - 1, 1)
            covs[i] += self.reg * np.eye(d)
        self.covs_ = covs

        self.cov_inv_ = np.array([np.linalg.inv(c) for c in covs])
        # slogdet is numerically stabler than log(det)
        _, log_dets   = zip(*(np.linalg.slogdet(c) for c in covs))
        self.log_det_ = np.array(log_dets)

    def _discriminant(self, X: NDArray) -> NDArray:
        assert (self.means_ is not None and self.cov_inv_ is not None
                and self.log_det_ is not None and self.log_priors_ is not None)
        n  = len(X)
        K  = len(self.means_)
        out = np.zeros((n, K))
        for k in range(K):
            diff  = X - self.means_[k]                                  # (n, d)
            mahal = np.einsum("ni,ij,nj->n", diff, self.cov_inv_[k], diff)
            out[:, k] = -0.5 * self.log_det_[k] - 0.5 * mahal + self.log_priors_[k]
        return out


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA

    print("Smoke test — LDA / QDA\n" + "-" * 40)
    for n_cls, label in [(2, "binary"), (4, "multinomial")]:
        # Avoid redundant features so sklearn QDA doesn't refuse a singular
        # covariance; our QDA tolerates it via `reg`.
        X, y = make_classification(
            n_samples=800, n_features=8, n_informative=8, n_redundant=0,
            n_classes=n_cls, n_clusters_per_class=1, class_sep=1.2,
            random_state=42,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

        for name, ours, sk in [
            ("LDA", LDA(), SkLDA()),
            ("QDA", QDA(), SkQDA(store_covariance=True)),
        ]:
            ours.fit(X_tr, y_tr); sk.fit(X_tr, y_tr)
            acc_ours = accuracy_score(y_te, ours.predict(X_te))
            acc_sk   = accuracy_score(y_te, sk.predict(X_te))
            print(f"{name} [{label:>11}]: ours={acc_ours:.4f}   sklearn={acc_sk:.4f}")


if __name__ == "__main__":
    main()
