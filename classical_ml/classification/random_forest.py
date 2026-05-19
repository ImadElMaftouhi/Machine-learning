"""Random Forest — bagged decision trees with feature randomness.

Algorithm:
    For each of `n_estimators` trees:
        1. Bootstrap sample of size n (with replacement).
        2. Pick a random subset of `max_features` columns.
        3. Fit a DecisionTree on (rows, cols) subview.
    Predict by soft voting (average per-tree predict_proba) or hard voting
    (majority of per-tree predict).

Design notes
------------
Feature subsampling is **per tree**, not per split. The classic Breiman RF
samples features at each split, which would require modifying DecisionTree
internals. Per-tree sampling is pedagogically clearer and keeps DecisionTree
untouched at the cost of slightly less diversity between trees.

Feature importances are computed as the average impurity-decrease attributed
to each feature across all trees, normalized to sum to 1.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

try:
    from ._utils import check_random_state
    from .decision_tree import DecisionTree, Node
except ImportError:
    from _utils import check_random_state                # type: ignore[no-redef]
    from decision_tree import DecisionTree, Node         # type: ignore[no-redef]


MaxFeaturesT = Union[Literal["sqrt", "log2"], int, float, None]


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: Literal["entropy", "gini"] = "gini",
        max_features: MaxFeaturesT = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

        # Set in fit()
        self.classes_: NDArray | None = None
        self.n_features_: int | None = None
        self.estimators_: list[DecisionTree] = []
        self.feature_indices_: list[NDArray] = []
        self.bootstrap_indices_: list[NDArray] = []
        self.oob_score_: float | None = None
        self.feature_importances_: NDArray | None = None

    def __repr__(self) -> str:
        return (f"RandomForest(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, criterion={self.criterion!r}, "
                f"max_features={self.max_features!r})")

    # ---------------- helpers ----------------

    def _resolve_max_features(self, d: int) -> int:
        mf = self.max_features
        if mf is None:
            return d
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(d)))
            if mf == "log2":
                return max(1, int(np.log2(d)))
            raise ValueError(
                f"max_features str must be 'sqrt' or 'log2'. Got {mf!r}."
            )
        if isinstance(mf, int):
            return max(1, min(mf, d))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        raise ValueError(f"Unknown max_features type: {type(mf).__name__}")

    def _align_proba(self, proba: NDArray, tree_classes: NDArray) -> NDArray:
        """Map a tree's per-class proba (might lack some classes) to global classes_."""
        assert self.classes_ is not None
        n = proba.shape[0]
        K = len(self.classes_)
        aligned = np.zeros((n, K))
        for i, c in enumerate(tree_classes):
            j = int(np.searchsorted(self.classes_, c))
            aligned[:, j] = proba[:, i]
        return aligned

    # ---------------- fit ----------------

    def fit(self, X: NDArray, y: NDArray) -> "RandomForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.n_features_ = d
        m = self._resolve_max_features(d)
        K = len(self.classes_)

        rng = check_random_state(self.random_state)

        self.estimators_ = []
        self.feature_indices_ = []
        self.bootstrap_indices_ = []

        oob_sum   = np.zeros((n, K)) if (self.oob_score and self.bootstrap) else None
        oob_count = np.zeros(n)      if (self.oob_score and self.bootstrap) else None

        for _ in range(self.n_estimators):
            if self.bootstrap:
                row_idx = rng.integers(0, n, size=n)
            else:
                row_idx = np.arange(n)
            feat_idx = rng.choice(d, m, replace=False)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
            ).fit(X[row_idx][:, feat_idx], y[row_idx])

            self.estimators_.append(tree)
            self.feature_indices_.append(feat_idx)
            self.bootstrap_indices_.append(row_idx)

            if oob_sum is not None and oob_count is not None:
                oob_rows = np.setdiff1d(np.arange(n), row_idx, assume_unique=False)
                if len(oob_rows) > 0:
                    proba = tree.predict_proba(X[oob_rows][:, feat_idx])
                    assert tree.classes_ is not None
                    oob_sum[oob_rows]   += self._align_proba(proba, tree.classes_)
                    oob_count[oob_rows] += 1

        if oob_sum is not None and oob_count is not None:
            valid = oob_count > 0
            if valid.any():
                pred = self.classes_[np.argmax(oob_sum[valid], axis=1)]
                self.oob_score_ = float(np.mean(pred == y[valid]))

        self._compute_feature_importances()
        return self

    # ---------------- predict ----------------

    def predict_proba(self, X: NDArray) -> NDArray:
        """Soft-voted probabilities.

        Binary → (n,) P(y = classes_[1])
        Multi  → (n, K)
        """
        assert self.classes_ is not None, "Call fit() before predict_proba()."
        X = np.asarray(X, dtype=float)
        n = len(X)
        K = len(self.classes_)
        avg = np.zeros((n, K))
        for tree, feat_idx in zip(self.estimators_, self.feature_indices_):
            assert tree.classes_ is not None
            proba = tree.predict_proba(X[:, feat_idx])
            avg  += self._align_proba(proba, tree.classes_)
        avg /= len(self.estimators_)
        return avg[:, 1] if K == 2 else avg

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None, "Call fit() before predict()."
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            return self.classes_[(proba >= 0.5).astype(int)]
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: NDArray, y: NDArray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ---------------- feature importances ----------------

    def _compute_feature_importances(self) -> None:
        """Average impurity decrease per feature, normalized.

        Each split node contributes `(n_samples_at_node / n_total) * decrease`
        to its splitting feature. We then map per-tree feature indices back to
        the original feature space, sum across trees, and normalize.
        """
        assert self.n_features_ is not None
        importances = np.zeros(self.n_features_)
        for tree, feat_idx in zip(self.estimators_, self.feature_indices_):
            assert tree.root_ is not None
            n_total = max(tree.root_.n_samples, 1)
            sub_imp = np.zeros(len(feat_idx))
            self._walk_importance(tree.root_, sub_imp, n_total)
            for sub, orig in enumerate(feat_idx):
                importances[orig] += sub_imp[sub]
        total = importances.sum()
        if total > 0:
            importances /= total
        self.feature_importances_ = importances

    def _walk_importance(self, node: Node | None, out: NDArray, n_total: int) -> None:
        if node is None or node.is_leaf:
            return
        assert node.feature is not None and node.left is not None and node.right is not None
        n_node  = node.n_samples
        n_left  = node.left.n_samples
        n_right = node.right.n_samples
        children_imp = (n_left * node.left.impurity + n_right * node.right.impurity) \
                       / max(n_node, 1)
        decrease = node.impurity - children_imp
        out[node.feature] += decrease * (n_node / max(n_total, 1))
        self._walk_importance(node.left,  out, n_total)
        self._walk_importance(node.right, out, n_total)


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier as SkRF

    print("Smoke test — RandomForest\n" + "-" * 40)
    for n_cls, label in [(2, "binary"), (4, "multinomial")]:
        X, y = make_classification(
            n_samples=800, n_features=10, n_informative=8, n_redundant=0,
            n_classes=n_cls, n_clusters_per_class=1, class_sep=1.2,
            random_state=42,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

        ours = RandomForest(n_estimators=50, max_depth=8,
                            oob_score=True).fit(X_tr, y_tr)
        sk = SkRF(n_estimators=50, max_depth=8, oob_score=True,
                  random_state=42).fit(X_tr, y_tr)
        acc_ours = accuracy_score(y_te, ours.predict(X_te))
        acc_sk   = accuracy_score(y_te, sk.predict(X_te))
        print(f"[{label:>11}]: ours={acc_ours:.4f} (oob={ours.oob_score_:.4f})   "
              f"sklearn={acc_sk:.4f} (oob={sk.oob_score_:.4f})")


if __name__ == "__main__":
    main()
