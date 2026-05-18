"""
Naive Bayes classifier — three variants.

Core idea: apply Bayes' theorem and assume features are conditionally
independent given the class label (the "naive" assumption).

    P(y | x) ∝ P(y) · ∏ P(x_j | y)
                prior   likelihood per feature

Working in log-space avoids underflow when multiplying many small probabilities.

Variants
--------
gaussian    — continuous features; models P(x_j | y) as a Gaussian.
              Best fit for real-valued data (e.g. make_classification).

multinomial — non-negative integer counts; models P(x_j | y) as a Multinomial.
              Typical for bag-of-words text classification.

bernoulli   — binary features (0/1); models P(x_j | y) as a Bernoulli.
              Useful when features represent presence/absence.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


# ─── shared log-sum-exp ───────────────────────────────────────────────────────

def _logsumexp(log_probs: NDArray) -> NDArray:
    """
    Numerically stable log( Σ exp(a_i) ) along axis=1.

    Trick: factor out max to keep exp() arguments near zero.
        log Σ exp(a_i) = max(a) + log Σ exp(a_i − max(a))
    """
    max_log = log_probs.max(axis=1, keepdims=True)
    return max_log + np.log(np.exp(log_probs - max_log).sum(axis=1, keepdims=True))


# ─── Gaussian Naive Bayes ─────────────────────────────────────────────────────

class GaussianNB:
    """
    Assumes P(x_j | y=k) = N(μ_jk, σ²_jk).

    Parameters computed in fit:
        priors_  (K,)    log P(y=k)
        means_   (K, d)  per-class, per-feature mean
        vars_    (K, d)  per-class, per-feature variance
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        var_smoothing : added to every variance to prevent division by zero
                        when a feature has zero variance within a class.
        """
        self.var_smoothing = var_smoothing

        self.classes_: NDArray | None = None
        self.log_priors_: NDArray | None = None
        self.means_: NDArray | None = None
        self.vars_:  NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> "GaussianNB":
        n, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        self.means_      = np.zeros((K, d))
        self.vars_       = np.zeros((K, d))
        self.log_priors_ = np.zeros(K)

        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.means_[i]      = X_c.mean(axis=0)
            self.vars_[i]       = X_c.var(axis=0) + self.var_smoothing
            self.log_priors_[i] = np.log(len(X_c) / n)

        return self

    def _log_likelihood(self, X: NDArray) -> NDArray:
        """
        Vectorized log P(x | y=k) for all samples and classes at once.

        log N(x_j; μ, σ²) = -½ log(2π σ²) - (x_j − μ)² / (2σ²)

        Shapes:
            X                        (n, d)
            means_, vars_            (K, d)
            X[:, None, :] - means_   (n, K, d)  ← broadcasting
            output                   (n, K)
        """
        assert self.means_ is not None and self.vars_ is not None

        # Constant term: -½ Σ_j log(2π σ²_jk)  →  (K,)
        log_const = -0.5 * np.log(2 * np.pi * self.vars_).sum(axis=1)

        # Quadratic term: -½ Σ_j (x_j - μ_jk)² / σ²_jk  →  (n, K)
        diff_sq   = (X[:, None, :] - self.means_[None, :, :]) ** 2   # (n, K, d)
        log_quad  = -0.5 * (diff_sq / self.vars_[None, :, :]).sum(axis=2)

        return log_const + log_quad   # (n, K)

    def predict_log_proba(self, X: NDArray) -> NDArray:
        """Unnormalized log posterior: log P(y=k) + log P(x | y=k)  →  (n, K)."""
        assert self.log_priors_ is not None
        return self._log_likelihood(X) + self.log_priors_

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Normalized posterior probabilities via log-sum-exp.
        Returns (n,) for binary, (n, K) for multi-class.
        """
        assert self.classes_ is not None
        log_post = self.predict_log_proba(X)
        log_Z    = _logsumexp(log_post)            # (n, 1)
        proba    = np.exp(log_post - log_Z)        # (n, K)
        return proba[:, 1] if len(self.classes_) == 2 else proba

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __repr__(self) -> str:
        return f"GaussianNB(var_smoothing={self.var_smoothing})"


# ─── Multinomial Naive Bayes ──────────────────────────────────────────────────

class MultinomialNB:
    """
    Assumes features are non-negative counts and P(x_j | y=k) is Multinomial.

    Estimated as a smoothed conditional probability (Laplace smoothing):
        P(x_j | y=k) = (count(x_j, k) + α) / (count(*, k) + α · d)

    Parameters
    ----------
    alpha : smoothing parameter (1.0 = Laplace, 0 = no smoothing)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

        self.classes_: NDArray | None = None
        self.log_priors_:      NDArray | None = None
        self.log_cond_probs_:  NDArray | None = None   # (K, d)

    def fit(self, X: NDArray, y: NDArray) -> "MultinomialNB":
        n, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        self.log_priors_     = np.zeros(K)
        self.log_cond_probs_ = np.zeros((K, d))

        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.log_priors_[i] = np.log(len(X_c) / n)

            # feature counts per class, smoothed
            counts = X_c.sum(axis=0) + self.alpha           # (d,)
            self.log_cond_probs_[i] = np.log(counts / counts.sum())

        return self

    def predict_log_proba(self, X: NDArray) -> NDArray:
        """
        log P(x | y=k) = Σ_j x_j · log P(x_j | y=k)   →  (n, K)
        """
        assert self.log_priors_ is not None
        assert self.log_cond_probs_ is not None
        return X @ self.log_cond_probs_.T + self.log_priors_   # (n, K)

    def predict_proba(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None
        log_post = self.predict_log_proba(X)
        proba    = np.exp(log_post - _logsumexp(log_post))
        return proba[:, 1] if len(self.classes_) == 2 else proba

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __repr__(self) -> str:
        return f"MultinomialNB(alpha={self.alpha})"


# ─── Bernoulli Naive Bayes ────────────────────────────────────────────────────

class BernoulliNB:
    """
    Assumes features are binary (0/1) and P(x_j | y=k) is Bernoulli.

        P(x_j | y=k) = p_jk^x_j · (1 − p_jk)^(1 − x_j)

    Non-binary inputs are binarized at `threshold` during predict.

    Parameters
    ----------
    alpha     : Laplace smoothing
    threshold : binarization cutoff (features > threshold become 1)
    """

    def __init__(self, alpha: float = 1.0, threshold: float = 0.0):
        self.alpha     = alpha
        self.threshold = threshold

        self.classes_: NDArray | None = None
        self.log_priors_: NDArray | None = None
        self.log_prob_:   NDArray | None = None   # (K, d)  log P(x_j=1 | y=k)
        self.log_neg_prob_: NDArray | None = None # (K, d)  log P(x_j=0 | y=k)

    def fit(self, X: NDArray, y: NDArray) -> "BernoulliNB":
        X = (X > self.threshold).astype(float)
        n, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        self.log_priors_  = np.zeros(K)
        self.log_prob_    = np.zeros((K, d))
        self.log_neg_prob_= np.zeros((K, d))

        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            n_c = len(X_c)
            self.log_priors_[i]   = np.log(n_c / n)

            # Laplace-smoothed P(x_j = 1 | y=k)
            p = (X_c.sum(axis=0) + self.alpha) / (n_c + 2 * self.alpha)
            self.log_prob_[i]     = np.log(p)
            self.log_neg_prob_[i] = np.log(1.0 - p)

        return self

    def predict_log_proba(self, X: NDArray) -> NDArray:
        """
        log P(x | y=k) = Σ_j [x_j · log p_jk + (1−x_j) · log(1−p_jk)]
                       = Σ_j log(1−p_jk) + Σ_j x_j · log(p_jk / (1−p_jk))
        """
        assert self.log_priors_   is not None
        assert self.log_prob_     is not None
        assert self.log_neg_prob_ is not None

        X_bin = (X > self.threshold).astype(float)
        # log-odds term summed over features where x_j = 1
        log_odds = self.log_prob_ - self.log_neg_prob_     # (K, d)
        log_lh   = (self.log_neg_prob_.sum(axis=1)         # (K,)  baseline
                    + X_bin @ log_odds.T)                  # (n, K) contribution
        return log_lh + self.log_priors_

    def predict_proba(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None
        log_post = self.predict_log_proba(X)
        proba    = np.exp(log_post - _logsumexp(log_post))
        return proba[:, 1] if len(self.classes_) == 2 else proba

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __repr__(self) -> str:
        return f"BernoulliNB(alpha={self.alpha}, threshold={self.threshold})"
