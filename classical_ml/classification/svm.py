"""Support Vector Machine — two trainers under one class.

Linear path (`kernel="linear"`):
    Minimize J(w, b) = ½‖w‖² + (C/n)·Σᵢ max(0, 1 − yᵢ(w·xᵢ + b))
    via batch sub-gradient descent. ~80 lines, fully transparent.

RBF path (`kernel="rbf"`):
    Solve the dual via Platt's *simplified SMO* (Andrew Ng CS229 version):
        max_α  Σᵢ αᵢ − ½ Σᵢⱼ yᵢ yⱼ αᵢ αⱼ K(xᵢ, xⱼ)
        s.t.   0 ≤ αᵢ ≤ C,    Σᵢ αᵢ yᵢ = 0
    Iterate: for each i with KKT violation, pick a random j ≠ i, analytically
    optimize the 2-variable subproblem (α_i, α_j), update b. Continue until
    `max_passes` consecutive sweeps make no changes.

Multi-class via OvR (one binary SVM per class). `predict_proba` is
sigmoid/softmax of the decision function — *uncalibrated*. Platt scaling
intentionally omitted.

The RBF Gram matrix is O(n²); experiments should cap training-set size
around 3000 samples.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

try:
    from ._utils import OvRClassifier, check_random_state, sigmoid
except ImportError:
    from _utils import OvRClassifier, check_random_state, sigmoid  # type: ignore[no-redef]


GammaT = Union[float, Literal["scale"]]


class SVM:
    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "rbf"] = "linear",
        gamma: GammaT = "scale",
        lr: float = 0.01,
        n_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int | None = 42,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Set in fit()
        self.classes_: NDArray | None = None
        # Linear path
        self.weights_: NDArray | None = None
        self.bias_: float = 0.0
        self.losses_: list[float] = []
        # Kernel path
        self.alphas_: NDArray | None = None
        self.support_: NDArray | None = None
        self.support_vectors_: NDArray | None = None
        self.support_y_: NDArray | None = None
        self.gamma_: float = 0.0
        # OvR for multi-class
        self._ovr: OvRClassifier | None = None

    def __repr__(self) -> str:
        return (f"SVM(C={self.C}, kernel={self.kernel!r}, "
                f"gamma={self.gamma!r})")

    # ---------------- Public API ----------------

    def fit(self, X: NDArray, y: NDArray) -> "SVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("SVM requires at least 2 classes.")
        if K == 2:
            self._fit_binary(X, y)
        else:
            self._ovr = OvRClassifier(base_factory=self._make_binary_clone)
            self._ovr.fit(X, y)
        return self

    def decision_function(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        if self._ovr is not None:
            return self._ovr.decision_function(X)
        if self.kernel == "linear":
            assert self.weights_ is not None, "Call fit() before decision_function()."
            return X @ self.weights_ + self.bias_
        return self._kernel_decision(X)

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None, "Call fit() before predict()."
        if self._ovr is not None:
            return self._ovr.predict(X)
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]

    def predict_proba(self, X: NDArray) -> NDArray:
        """Uncalibrated sigmoid/softmax of decision scores."""
        if self._ovr is not None:
            return self._ovr.predict_proba(X)
        return sigmoid(self.decision_function(X))

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ---------------- Internals ----------------

    def _make_binary_clone(self) -> "SVM":
        return SVM(
            C=self.C, kernel=self.kernel, gamma=self.gamma,
            lr=self.lr, n_iter=self.n_iter, tol=self.tol,
            random_state=self.random_state,
        )

    def _fit_binary(self, X: NDArray, y: NDArray) -> None:
        assert self.classes_ is not None
        # Map labels → {-1, +1}: classes_[0] → -1, classes_[1] → +1.
        y_signed = np.where(y == self.classes_[1], 1, -1).astype(float)
        if self.kernel == "linear":
            self._fit_linear(X, y_signed)
        else:
            self._resolve_gamma(X)
            self._fit_smo(X, y_signed)

    def _resolve_gamma(self, X: NDArray) -> None:
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                var = X.var()
                self.gamma_ = 1.0 / max(X.shape[1] * float(var), 1e-12)
            else:
                raise ValueError(f"Unknown gamma string: {self.gamma!r}")
        else:
            self.gamma_ = float(self.gamma)

    # --- Linear: primal sub-gradient descent ---

    def _fit_linear(self, X: NDArray, y_signed: NDArray) -> None:
        n, d = X.shape
        self.weights_ = np.zeros(d)
        self.bias_ = 0.0
        self.losses_ = []

        for it in range(self.n_iter):
            margins  = y_signed * (X @ self.weights_ + self.bias_)
            violated = margins < 1.0

            hinge = float(np.maximum(0.0, 1.0 - margins).sum())
            loss  = 0.5 * float(self.weights_ @ self.weights_) + (self.C / n) * hinge
            self.losses_.append(loss)

            # Sub-gradient
            grad_w = self.weights_ - (self.C / n) * (y_signed[violated] @ X[violated])
            grad_b = -(self.C / n) * float(y_signed[violated].sum())

            self.weights_ -= self.lr * grad_w
            self.bias_    -= self.lr * grad_b

            # Convergence check
            if it > 0 and abs(self.losses_[-2] - self.losses_[-1]) < self.tol:
                break

    # --- Kernel: RBF Gram and decision function ---

    def _rbf_kernel(self, X1: NDArray, X2: NDArray) -> NDArray:
        """K(x, z) = exp(-γ ‖x - z‖²) — vectorized."""
        sq1 = (X1 ** 2).sum(axis=1, keepdims=True)
        sq2 = (X2 ** 2).sum(axis=1)
        d2  = sq1 + sq2 - 2.0 * (X1 @ X2.T)
        return np.exp(-self.gamma_ * np.clip(d2, 0.0, None))

    def _kernel_decision(self, X: NDArray) -> NDArray:
        assert (self.support_vectors_ is not None and self.alphas_ is not None
                and self.support_y_ is not None)
        if len(self.support_vectors_) == 0:
            return np.full(len(X), self.bias_)
        K = self._rbf_kernel(X, self.support_vectors_)        # (n_test, n_sv)
        return K @ (self.alphas_ * self.support_y_) + self.bias_

    # --- RBF: simplified SMO ---

    def _fit_smo(self, X: NDArray, y_signed: NDArray) -> None:
        n = len(X)
        K = self._rbf_kernel(X, X)                       # (n, n) — cap n in callers
        alphas = np.zeros(n)
        b = 0.0
        rng = check_random_state(self.random_state)
        max_passes = max(5, self.n_iter // 50)
        passes = 0

        while passes < max_passes:
            num_changed = 0
            for i in range(n):
                # E_i = f(x_i) - y_i
                f_i = float((alphas * y_signed) @ K[:, i]) + b
                E_i = f_i - y_signed[i]

                # KKT violation check
                if ((y_signed[i] * E_i < -self.tol and alphas[i] < self.C) or
                    (y_signed[i] * E_i >  self.tol and alphas[i] > 0)):

                    # Pick j ≠ i uniformly at random
                    j = int(rng.integers(0, n))
                    while j == i:
                        j = int(rng.integers(0, n))

                    f_j = float((alphas * y_signed) @ K[:, j]) + b
                    E_j = f_j - y_signed[j]

                    a_i_old, a_j_old = alphas[i], alphas[j]

                    # Box-constraint bounds for α_j
                    if y_signed[i] != y_signed[j]:
                        L = max(0.0,        alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0.0,    alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    if L == H:
                        continue

                    # Second derivative of the dual along the constraint line
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update α_j, then clip to [L, H]
                    alphas[j] = a_j_old - y_signed[j] * (E_i - E_j) / eta
                    alphas[j] = float(np.clip(alphas[j], L, H))

                    if abs(alphas[j] - a_j_old) < 1e-5:
                        continue

                    # Update α_i to preserve the equality constraint
                    alphas[i] = a_i_old + y_signed[i] * y_signed[j] * (a_j_old - alphas[j])

                    # Update b — choose interior endpoint if available
                    b1 = (b - E_i
                          - y_signed[i] * (alphas[i] - a_i_old) * K[i, i]
                          - y_signed[j] * (alphas[j] - a_j_old) * K[i, j])
                    b2 = (b - E_j
                          - y_signed[i] * (alphas[i] - a_i_old) * K[i, j]
                          - y_signed[j] * (alphas[j] - a_j_old) * K[j, j])

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0

        # Extract support vectors: α > 0
        sv_mask = alphas > 1e-7
        self.alphas_          = alphas[sv_mask]
        self.support_         = np.where(sv_mask)[0]
        self.support_vectors_ = X[sv_mask]
        self.support_y_       = y_signed[sv_mask]
        self.bias_            = float(b)


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    print("Smoke test — SVM\n" + "-" * 40)
    for n_cls, label in [(2, "binary"), (3, "multinomial")]:
        X, y = make_classification(
            n_samples=400, n_features=6, n_informative=5,
            n_redundant=0, n_classes=n_cls, n_clusters_per_class=1,
            class_sep=1.2, random_state=42,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

        for kernel in ("linear", "rbf"):
            ours = SVM(C=1.0, kernel=kernel, lr=0.01, n_iter=300).fit(X_tr, y_tr)
            sk   = SVC(C=1.0, kernel=kernel, gamma="scale",
                       random_state=42).fit(X_tr, y_tr)
            a_o  = accuracy_score(y_te, ours.predict(X_te))
            a_s  = accuracy_score(y_te, sk.predict(X_te))
            extra = ""
            if kernel == "rbf" and ours._ovr is None and ours.support_vectors_ is not None:
                extra = f"   SVs={len(ours.support_vectors_)}/{len(X_tr)}"
            print(f"{label:>11} {kernel:>6}: ours={a_o:.4f}   sklearn={a_s:.4f}{extra}")


if __name__ == "__main__":
    main()
