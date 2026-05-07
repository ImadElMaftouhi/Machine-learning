"""
Polynomial Regression — extends linear regression by augmenting the feature
space with polynomial terms of the original inputs.

For X with p columns and polynomial degree d, the design matrix Phi contains
all monomials of total degree <= d. The model remains *linear in beta*:

    y = beta_0 + sum_k beta_k * monomial_k(x) + eps

so the full OLS / Ridge theory (normal equations, Cholesky / QR / SVD,
gradient descent) applies directly to Phi.

Trade-off: |Phi|_cols = C(p+d, d) grows fast. Forming Phi explicitly is fine
when this is moderate; otherwise switch to a polynomial kernel (Kernel Ridge
Regression, doc §12.3) and work in the dual space.
"""

from itertools import combinations, combinations_with_replacement

import numpy as np
import scipy.linalg as la


VALID_METHODS = ("cholesky", "qr", "svd", "gradient_descent")


class PolynomialRegression:
    def __init__(self, degree: int = 2, interaction_only: bool = False,
                 include_bias: bool = True, penalty: float = 0.0,
                 method: str = "cholesky", lr: float = 0.01,
                 n_iter: int = 2000, tol: float = 1e-6):
        if degree < 1:
            raise ValueError(f"degree must be >= 1. Got {degree}.")
        if penalty < 0:
            raise ValueError(f"penalty must be >= 0. Got {penalty}.")
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}. Got {method!r}.")

        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.penalty = penalty
        self.method = method
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.feature_indices_ = None   # list of tuples; () represents the bias column
        self.n_input_features_ = None
        self.loss_history = []

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_feature_indices(self, p: int):
        """Build the list of monomial index tuples once (during fit)."""
        indices = []
        if self.include_bias:
            indices.append(())  # the constant 1 column
        for d in range(1, self.degree + 1):
            combos = (combinations(range(p), d) if self.interaction_only
                      else combinations_with_replacement(range(p), d))
            indices.extend(combos)
        return indices

    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Map X (n, p) -> Phi (n, m) using stored monomial indices."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape

        if self.feature_indices_ is None:
            self.n_input_features_ = p
            self.feature_indices_ = self._build_feature_indices(p)
        elif p != self.n_input_features_:
            raise ValueError(
                f"X has {p} features, but model was fitted with "
                f"{self.n_input_features_}."
            )

        cols = []
        for idx in self.feature_indices_:
            if len(idx) == 0:
                cols.append(np.ones(n))
            else:
                cols.append(np.prod(X[:, list(idx)], axis=1))
        return np.column_stack(cols)

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------
    def fit(self, X, y):
        self.loss_history = []
        self.feature_indices_ = None  # force rebuild for new X shape

        Phi = self._polynomial_features(X)
        y = np.asarray(y, dtype=float).flatten()
        n, m = Phi.shape

        # Don't regularize the bias term (it lives at column 0 when include_bias=True).
        I = np.eye(m)
        if self.include_bias:
            I[0, 0] = 0.0

        if self.method == "cholesky":
            # (Phi^T Phi + lambda*I) beta = Phi^T y, solved via Cholesky (SPD).
            A = Phi.T @ Phi + self.penalty * I
            b = Phi.T @ y
            self.coef_ = la.solve(A, b, assume_a="pos")

        elif self.method == "qr":
            # Stack the ridge problem as augmented OLS, then thin QR.
            sqrt_lam = np.sqrt(self.penalty)
            Phi_stack = np.vstack([Phi, sqrt_lam * I])
            y_stack = np.concatenate([y, np.zeros(m)])
            Q, R = np.linalg.qr(Phi_stack, mode="reduced")
            self.coef_ = la.solve_triangular(R, Q.T @ y_stack)

        elif self.method == "svd":
            # SVD of Phi yields the classic ridge filter d_j = s_j / (s_j^2 + lambda).
            U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
            d_filter = s / (s ** 2 + self.penalty)
            self.coef_ = Vt.T @ (d_filter * (U.T @ y))

        elif self.method == "gradient_descent":
            # Safe step size: 1 / L where L = 2*(||Phi||_F^2 / n + lambda).
            L = 2.0 * (np.linalg.norm(Phi, "fro") ** 2 / n + self.penalty)
            lr = min(self.lr, 1.0 / L)
            self.coef_ = np.zeros(m)
            for _ in range(self.n_iter):
                residual = Phi @ self.coef_ - y
                grad = (2.0 / n) * (Phi.T @ residual) + 2.0 * self.penalty * (I @ self.coef_)
                self.coef_ -= lr * grad
                loss = (residual @ residual) / n + self.penalty * (self.coef_ @ (I @ self.coef_))
                if not np.isfinite(loss):
                    break
                self.loss_history.append(loss)
                if len(self.loss_history) > 10 and \
                   abs(self.loss_history[-1] - self.loss_history[-10]) < self.tol:
                    break

        assert self.coef_ is not None
        self.intercept_ = float(self.coef_[0]) if self.include_bias else 0.0
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._polynomial_features(X) @ self.coef_

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_feature_names(self, input_names=None):
        """Return human-readable monomial labels matching coef_."""
        if self.feature_indices_ is None or self.n_input_features_ is None:
            raise ValueError("Model not fitted yet.")
        if input_names is None:
            input_names = [f"x{i}" for i in range(self.n_input_features_)]
        names = []
        for idx in self.feature_indices_:
            if len(idx) == 0:
                names.append("1")
                continue
            counts = {}
            for j in idx:
                counts[j] = counts.get(j, 0) + 1
            parts = [f"{input_names[j]}^{k}" if k > 1 else input_names[j]
                     for j, k in sorted(counts.items())]
            names.append(" * ".join(parts))
        return names

    def __repr__(self):
        return (f"PolynomialRegression(degree={self.degree}, "
                f"interaction_only={self.interaction_only}, "
                f"include_bias={self.include_bias}, "
                f"penalty={self.penalty}, method={self.method!r})")
