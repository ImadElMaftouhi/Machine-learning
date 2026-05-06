## Classical Machine Learning Roadmap

### Phase 1: Core Mathematical Prerequisites

Study these in parallel with Phase 2.

| Topic | What you need | Where it shows up |
|---|---|---|
| Linear algebra | Vectors, matrices, dot products, eigenvalues, SVD | PCA, linear regression, SVM |
| Probability & statistics | Bayes' rule, MLE, bias-variance tradeoff | Naive Bayes, GMM, model evaluation |
| Calculus | Gradients, partial derivatives, convexity | Gradient descent, loss functions |

Work through one concept at a time. Write a short NumPy exercise for each before moving on.

---

### Phase 2: `regression/`

**Goal**: understand how supervised learning works through the simplest output space — continuous values.

#### Algorithms to implement

| File | Algorithm | Notes |
|---|---|---|
| `linear_regression.py` | Ordinary least squares | Implement both closed-form (normal equations) and gradient descent. Compare results. |
| `ridge_regression.py` | Ridge (L2 regularization) | Extend your OLS implementation. Observe how λ controls overfitting. |
| `polynomial_regression.py` | Polynomial features + linear regression | Use NumPy to construct feature matrix manually. |

#### Notebooks

| File | Purpose |
|---|---|
| `regression_scratch_vs_sklearn.ipynb` | Validate your implementations against scikit-learn on a synthetic dataset |
| `regression_real_world.ipynb` | Apply to a real dataset (e.g., California housing). Practice EDA, feature scaling, residual plots. |

#### Concepts to understand before moving on

- Why the normal equation breaks down at scale (matrix inversion cost)
- What regularization does geometrically
- How to read a residual plot

---

### Phase 3: `classification/`

**Goal**: learn the core supervised classification methods. Each one introduces a new idea.

#### Algorithms to implement

| File | Algorithm | Key idea introduced |
|---|---|---|
| `logistic_regression.py` | Logistic regression | Sigmoid, cross-entropy loss, decision boundary |
| `knn.py` | k-Nearest Neighbors | Nonparametric prediction, distance metrics, curse of dimensionality |
| `decision_tree.py` | Decision tree (CART) | Gini / entropy splitting, recursive partitioning, depth as regularization |
| `naive_bayes.py` | Gaussian Naive Bayes | Conditional independence assumption, generative modeling |
| `svm.py` | Linear SVM | Margin maximization, soft margin, hinge loss |

#### Notebooks

| File | Purpose |
|---|---|
| `binary_classification.ipynb` | Logistic regression vs decision tree on a binary tabular dataset (e.g., Titanic, credit fraud). Covers EDA, feature engineering, cross-validation. |
| `multiclass_classification.ipynb` | SVM and kNN on a multiclass problem (e.g., Iris, digit pixels). Covers confusion matrix, per-class metrics. |
| `classification_scratch_vs_sklearn.ipynb` | Validate all scratch implementations against scikit-learn equivalents. |

#### Concepts to understand before moving on

- Why logistic regression is a linear classifier despite using a nonlinear activation
- The difference between a parametric and nonparametric model
- What a kernel does to a feature space (intuition only — full derivation comes in Phase 6)

---

### Phase 4: `clustering/`

**Goal**: move into unsupervised learning. There are no labels — evaluation is more subtle.

#### Algorithms to implement

| File | Algorithm | Key idea introduced |
|---|---|---|
| `kmeans.py` | k-Means | Expectation-maximization flavor, initialization sensitivity (k-means++) |
| `hierarchical.py` | Agglomerative clustering | Linkage criteria (single, complete, Ward), dendrogram reading |
| `gmm.py` | Gaussian Mixture Model | Full EM algorithm, soft assignments, log-likelihood convergence |

#### Notebooks

| File | Purpose |
|---|---|
| `clustering_exploration.ipynb` | Apply all three methods to a synthetic 2D dataset. Visualize cluster boundaries and dendrograms. |
| `clustering_real_world.ipynb` | Apply to a real dataset. Use the Elbow method and silhouette score to choose k. |
| `clustering_as_preprocessing.ipynb` | Use cluster labels as features in a downstream classifier. Compare performance with and without. |

#### Concepts to understand before moving on

- Why k-means is sensitive to initialization and how k-means++ mitigates it
- The difference between hard and soft assignment
- What the EM algorithm is actually doing at each step

---

### Phase 5: `dimensionality_reduction/`

**Goal**: reduce feature space while preserving structure. Both for visualization and as a preprocessing step.

#### Algorithms to implement

| File | Algorithm | Key idea introduced |
|---|---|---|
| `pca.py` | Principal Component Analysis | Variance maximization, eigenvectors of the covariance matrix, reconstruction error |
| `lda.py` | Linear Discriminant Analysis | Class separability, supervised dimensionality reduction |

#### Notebooks

| File | Purpose |
|---|---|
| `pca_exploration.ipynb` | Apply PCA to a high-dimensional dataset (e.g., faces, digits). Plot explained variance ratio. Reconstruct from k components. |
| `pca_then_classify.ipynb` | PCA as preprocessing: reduce dimensions, then feed into a classifier. Compare accuracy vs. number of components retained. |
| `lda_vs_pca.ipynb` | Compare unsupervised (PCA) vs. supervised (LDA) reduction on the same dataset. |

#### Concepts to understand before moving on

- Why PCA finds orthogonal directions
- The difference between PCA (unsupervised) and LDA (supervised)
- How to choose the number of components (explained variance threshold vs. downstream task performance)

---

### Phase 6: Model Evaluation

This is not a separate folder — evaluation runs through every phase. But before starting projects, consolidate it explicitly.

#### Checklist

- [ ] Train-test split and k-fold cross-validation (stratified for imbalanced classes)
- [ ] Classification metrics: accuracy, precision, recall, F1, ROC-AUC — know when each is appropriate
- [ ] Regression metrics: MSE, MAE, R²
- [ ] Learning curves to diagnose underfitting vs. overfitting
- [ ] Hyperparameter tuning: grid search vs. random search
- [ ] Pipelines in scikit-learn: chain preprocessing + model + evaluation cleanly

Build a `model_selection.ipynb` notebook that demonstrates each of the above on a single dataset before you consider this phase done.

---

### Phase 7: End-to-End Projects

These live in the top-level `projects/` folder. Complete three before moving to deep learning.

| Project | Task | Techniques |
|---|---|---|
| Project 1 | Binary classification on tabular data (Titanic, credit fraud) | EDA, feature engineering, logistic regression vs. decision tree, cross-validation |
| Project 2 | Multiclass classification with high-dimensional features (handwritten digits) | PCA for visualization, SVM with RBF kernel, confusion matrix analysis |
| Project 3 | Unsupervised preprocessing for supervised task | Cluster labels as features, or PCA then classifier, full scikit-learn pipeline |

---

### Phase 8: Theory for Rigor (Ongoing)

These topics separate practitioners who copy code from those who debug failures. Work through one per week alongside projects.

- **Bias-variance decomposition** — derive it once on paper
- **No free lunch theorems** — understand what they actually claim (uniformity over all problems) and what they do not
- **VC dimension and PAC learning** — at least the intuitive meaning: what makes a function class learnable and how much data suffices
- **Convexity in loss functions** — why convex losses guarantee global optima and why non-convex ones do not
- **Kernel methods** — the full reproducing kernel Hilbert space story behind the SVM kernel trick

---

### Implementation Rules

- Scratch implementations: NumPy only, no scikit-learn internals
- Validation: always compare your scratch implementation to the scikit-learn equivalent before moving on
- Notebooks: one notebook per concept or experiment; extract reusable logic into `.py` modules
- No PyTorch or TensorFlow anywhere in this module

---

### Recommended Reference

Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (free from the authors).
Read chapters 1–9 in order. Skip matrix algebra derivations on first pass; return to them during Phase 8.

---

### Progress Tracker

| Phase | Folder | Status |
|---|---|---|
| 1 | Math prerequisites | ⬜ not started |
| 2 | `regression/` | ⬜ not started |
| 3 | `classification/` | ⬜ not started |
| 4 | `clustering/` | ⬜ not started |
| 5 | `dimensionality_reduction/` | ⬜ not started |
| 6 | Model evaluation | ⬜ not started |
| 7 | Projects | ⬜ not started |
| 8 | Theory | ⬜ ongoing |
