"""Decision Tree classifier — CART-style binary splits."""
"""TODO
Change the architecture of the file to account for all variant of decision tree

DecisionTreeBase    ← abstract: shared fit/predict skeleton + entropy/gini helpers
├── CART            ← binary splits, left/right Node
├── ID3             ← multi-way splits, dict-of-children Node  
└── C45             ← mixed binary/multi-way, gain ratio

"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Node:
    """A node in the decision tree.

    Internal nodes carry `feature`, `threshold`, `left`, `right`.
    Leaves carry `value` (predicted class) and `probas` (class distribution).
    """
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[int] = None
    probas: Optional[NDArray] = None
    n_samples: int = 0
    impurity: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, min_samples_leaf: int = 1, criterion: Literal["entropy", "gini"] = "entropy", min_impurity_decrease: float = 0.0,):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease

        # public fitted attributes
        self.root_: Optional[Node] = None
        self.classes_: Optional[NDArray] = None
        self.n_features_: Optional[int] = None

    def __repr__(self) -> str:
        return (f"DecisionTree(max_depth={self.max_depth}, "
                f"criterion={self.criterion!r})")

    # ---------------- Public API ----------------
    def fit(self, X: NDArray, y: NDArray) -> "DecisionTree":
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.root_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X: NDArray) -> NDArray:
        assert self.classes_ is not None, "Call fit() before predict()."
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(self, X: NDArray) -> NDArray:
        assert self.root_ is not None, "Call fit() before predict_proba()."
        X = np.asarray(X)
        return np.array([self._traverse(x, self.root_) for x in X])

    # ---------------- Tree construction ----------------
    def _grow_tree(self, X: NDArray, y: NDArray, depth: int) -> Node:
        n_samples = len(y)
        impurity = self._impurity(y)

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or impurity == 0.0
        ):
            return self._make_leaf(y, impurity, n_samples)

        split = self._best_split(X, y, parent_impurity=impurity)
        if split is None:
            return self._make_leaf(y, impurity, n_samples)

        feature, threshold, gain = split
        if gain < self.min_impurity_decrease:
            return self._make_leaf(y, impurity, n_samples)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask],  y[left_mask],  depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=feature, threshold=threshold,
            left=left, right=right,
            n_samples=n_samples, impurity=impurity,
        )

    def _best_split(
        self, X: NDArray, y: NDArray, parent_impurity: float,
    ) -> Optional[tuple[int, float, float]]:
        """Find (feature, threshold, gain) that maximizes impurity decrease."""
        n = len(y)
        best_gain = -1.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None

        for feature in range(X.shape[1]):
            column = X[:, feature]
            values = np.unique(column)
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = column <= threshold
                n_left = int(left_mask.sum())
                n_right = n - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                children_impurity = (
                    (n_left  / n) * self._impurity(y[left_mask]) +
                    (n_right / n) * self._impurity(y[~left_mask])
                )
                gain = parent_impurity - children_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)

        if best_feature is None or best_threshold is None:
            return None
        return best_feature, best_threshold, best_gain

    def _make_leaf(self, y: NDArray, impurity: float, n_samples: int) -> Node:
        assert self.classes_ is not None
        probas = np.array([np.mean(y == c) for c in self.classes_])
        value = self.classes_[int(np.argmax(probas))]
        return Node(
            value=value, probas=probas,
            n_samples=n_samples, impurity=impurity,
        )

    # ---------------- Impurity ----------------
    def _impurity(self, y: NDArray) -> float:
        if self.criterion == "entropy":
            return self._entropy(y)
        if self.criterion == "gini":
            return self._gini(y)
        raise ValueError(f"Unknown criterion: {self.criterion!r}")

    @staticmethod
    def _entropy(y: NDArray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _gini(y: NDArray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return float(1.0 - np.sum(p ** 2))

    # ---------------- Traversal ----------------
    def _traverse(self, x: NDArray, node: Node) -> NDArray:
        if node.is_leaf:
            assert node.probas is not None
            return node.probas
        assert node.feature is not None and node.threshold is not None
        assert node.left is not None and node.right is not None
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ---------------- Introspection ----------------
    def print_tree(self, node: Optional[Node] = None, depth: int = 0) -> None:
        if node is None:
            assert self.root_ is not None, "Call fit() before print_tree()."
            node = self.root_
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}leaf: class={node.value} "
                  f"(n={node.n_samples}, impurity={node.impurity:.3f})")
        else:
            print(f"{indent}X[{node.feature}] <= {node.threshold:.4f} "
                  f"(n={node.n_samples}, impurity={node.impurity:.3f})")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)


def main():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=1000, n_features=10,
        n_informative=6, n_redundant=2,
        n_classes=3, n_clusters_per_class=1,
        class_sep=1.0, flip_y=0.02,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    for criterion in ("entropy", "gini"):
        tree = DecisionTree(max_depth=5, criterion=criterion).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, tree.predict(X_te))
        print(f"{criterion:>7}: test accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
