# classification/logistic_regression.py
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, type: str = "binary"):
        if type not in ("binary", "multinomial", "ordinal"):
            raise ValueError(f"type must be 'binary', 'multinomial', or 'ordinal'. Got {type!r}.")
        self.type = type
        self.learning_rate = lr
        self.n_iter = n_iter

        # set by fit(); typed explicitly so the checker sees them post-fit
        self.weights:  NDArray[np.floating] | None = None
        self.bias:     NDArray[np.floating] | float | None = None
        self.thetas_:  NDArray[np.floating] | None = None
        self.classes_: NDArray[np.integer]  | None = None
        self.losses_:  list[float] = []

    def _check_fitted(self) -> None:
        if self.weights is None or self.classes_ is None:
            raise RuntimeError("Call fit() before predict.")

    # ── activations ──────────────────────────────────────────────────────────

    def _sigmoid(self, z):
        # Numerically stable: clip to avoid exp overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _softmax(self, z):
        # z : (n, K)  →  (n, K), rows sum to 1
        z_shifted = z - z.max(axis=1, keepdims=True)   # numerical stability
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _cumulative_probs(self, X):
        """
        Ordinal: K-1 cumulative P(y <= k | x) using proportional-odds model.
        Returns array of shape (n, K-1).
        """
        # logit = theta_k - w^T x  for each boundary k
        linear = X @ self.weights                        # (n,)
        return self._sigmoid(self.thetas_ - linear[:, None])   # (n, K-1)

    # ── fit helpers ───────────────────────────────────────────────────────────

    def _fit_binary(self, X, y):
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias    = 0.0

        for i in range(self.n_iter):
            y_hat = self._sigmoid(X @ self.weights + self.bias)
            loss  = -np.mean(
                y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9)
            )
            self.losses_.append(loss)

            err = y_hat - y
            self.weights -= self.learning_rate * (X.T @ err) / n
            self.bias    -= self.learning_rate * err.mean()

            if (i + 1) % max(1, self.n_iter // 10) == 0:
                print(f"[binary]  iter {i+1:>5}/{self.n_iter}  loss={loss:.4f}")

    def _fit_multinomial(self, X, y):
        assert self.classes_ is not None
        n, d = X.shape
        K = len(self.classes_)
        # one-hot encode
        Y = (y[:, None] == self.classes_[None, :]).astype(float)  # (n, K)

        self.weights = np.zeros((K, d))   # W : (K, d)
        self.bias    = np.zeros(K)        # b : (K,)

        for i in range(self.n_iter):
            assert self.weights is not None and self.bias is not None
            logits = X @ self.weights.T + self.bias   # (n, K)
            P      = self._softmax(logits)            # (n, K)

            loss = -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))
            self.losses_.append(loss)

            # gradient: dW (K, d), db (K,)
            delta = (P - Y) / n                       # (n, K)
            self.weights -= self.learning_rate * (delta.T @ X)
            self.bias    -= self.learning_rate * delta.sum(axis=0)

            if (i + 1) % max(1, self.n_iter // 10) == 0:
                print(f"[multinomial]  iter {i+1:>5}/{self.n_iter}  loss={loss:.4f}")

    def _fit_ordinal(self, X, y):
        """
        Proportional-odds model.
        One shared weight vector w, K-1 ordered thresholds theta_1 < ... < theta_{K-1}.
        Thresholds parameterised as theta_k = theta_1 + sum of softplus(delta_j) to
        enforce strict ordering without constraints.
        """
        assert self.classes_ is not None
        n, d = X.shape
        K = len(self.classes_)

        self.weights = np.zeros(d)
        # initialise thresholds spread around 0
        raw_thetas = np.linspace(-1.0, 1.0, K - 1)

        def _thetas_from_raw(raw):
            # raw[0] is theta_1; subsequent thresholds enforce ordering via softplus
            thresholds = np.empty(K - 1)
            thresholds[0] = raw[0]
            for k in range(1, K - 1):
                thresholds[k] = thresholds[k - 1] + np.log1p(np.exp(raw[k]))  # softplus
            return thresholds

        def _softplus_grad(raw_k):
            return 1.0 / (1.0 + np.exp(-raw_k))    # sigmoid(raw_k)

        for i in range(self.n_iter):
            self.thetas_ = _thetas_from_raw(raw_thetas)
            cum_p = self._cumulative_probs(X)        # (n, K-1)  P(y <= k)

            # class probabilities: p_k = P(y<=k) - P(y<=k-1)
            p_left  = np.hstack([cum_p, np.ones((n, 1))])    # (n, K)
            p_right = np.hstack([np.zeros((n, 1)), cum_p])   # (n, K)
            P = np.clip(p_left - p_right, 1e-9, 1.0)         # (n, K)

            # one-hot
            Y = (y[:, None] == self.classes_[None, :]).astype(float)

            loss = -np.mean(np.sum(Y * np.log(P), axis=1))
            self.losses_.append(loss)

            # Gradients via chain rule through cumulative probabilities
            # dL/d(cum_p_k) = -mean( Y_k/P_k - Y_{k+1}/P_{k+1} )
            dL_dcum = -(Y[:, :-1] / P[:, :-1] - Y[:, 1:] / P[:, 1:]) / n   # (n, K-1)

            # cum_p = sigmoid(theta_k - w^T x)  → derivative wrt argument is cum_p*(1-cum_p)
            d_sigmoid = cum_p * (1.0 - cum_p)   # (n, K-1)
            common = dL_dcum * d_sigmoid         # (n, K-1)

            # gradient wrt w: d(theta_k - w^T x)/dw = -x
            dw = -(common.sum(axis=1) @ X) / n    # ... sum over K-1, then over n
            dw = -common.sum(axis=1) @ X / n

            # gradient wrt each raw_theta
            d_raw = np.zeros(K - 1)
            # theta_k = theta_1 + softplus(raw_2) + ... so chain through ordering
            d_theta = common.mean(axis=0)          # (K-1,) gradient wrt thetas
            # propagate through the softplus reparameterisation
            for k in range(K - 2, -1, -1):
                d_raw[k] = d_theta[k]
                if k > 0:
                    d_raw[k] *= _softplus_grad(raw_thetas[k])
                # accumulate: theta_{k+1} depends on theta_k
                if k > 0:
                    d_theta[k - 1] += d_theta[k]

            self.weights  -= self.learning_rate * dw
            raw_thetas    -= self.learning_rate * d_raw

            if (i + 1) % max(1, self.n_iter // 10) == 0:
                print(f"[ordinal]  iter {i+1:>5}/{self.n_iter}  loss={loss:.4f}  "
                      f"thetas={np.round(self.thetas_, 3)}")

        self.thetas_ = _thetas_from_raw(raw_thetas)

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X, y):
        self.losses_ = []
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        if self.type == "binary":
            if len(self.classes_) != 2:
                raise ValueError("binary mode requires exactly 2 classes.")
            self._fit_binary(X, y)

        elif self.type == "multinomial":
            if len(self.classes_) < 2:
                raise ValueError("multinomial mode requires at least 2 classes.")
            self._fit_multinomial(X, y)

        elif self.type == "ordinal":
            if len(self.classes_) < 2:
                raise ValueError("ordinal mode requires at least 2 classes.")
            self._fit_ordinal(X, y)

        return self

    def predict_proba(self, X):
        """
        Returns:
          binary      → (n,)    P(y=1)
          multinomial → (n, K)  P(y=k) for each class
          ordinal     → (n, K)  P(y=k) recovered from cumulative probs
        """
        self._check_fitted()
        assert self.weights is not None and self.bias is not None

        if self.type == "binary":
            return self._sigmoid(X @ self.weights + self.bias)

        elif self.type == "multinomial":
            return self._softmax(X @ self.weights.T + self.bias)

        else:  # ordinal
            cum_p = self._cumulative_probs(X)              # (n, K-1)
            p_left  = np.hstack([cum_p, np.ones((len(X), 1))])
            p_right = np.hstack([np.zeros((len(X), 1)), cum_p])
            return np.clip(p_left - p_right, 0.0, 1.0)    # (n, K)

    def predict(self, X, threshold=0.5):
        """
        binary      → 0/1 using threshold
        multinomial → argmax of softmax probabilities
        ordinal     → argmax of class probabilities (respects ordering)
        """
        self._check_fitted()
        assert self.classes_ is not None

        if self.type == "binary":
            proba = self.predict_proba(X)
            assert proba is not None
            return (proba >= threshold).astype(int)

        else:   # multinomial and ordinal both use argmax
            proba = self.predict_proba(X)
            assert proba is not None
            return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))

    def __str__(self):
        return f"Logistic Regression Classifier ({self.type})"

    def __repr__(self):
        return (f"LogisticRegression(lr={self.learning_rate}, "
                f"n_iter={self.n_iter}, type={self.type!r})")