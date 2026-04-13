
## Classical Machine Learning Roadmap

### Phase 1: Core Mathematical Prerequisites

You cannot understand classical ML without three mathematical pillars. Study them in parallel with early algorithms, not as a separate boot camp.

- **Linear algebra**: Vectors, matrices, dot products, eigenvalues, singular value decomposition. Required for PCA, linear regression, support vector machines, and nearly every optimization step.
- **Probability and statistics**: Conditional probability, Bayes' rule, expectation, variance, maximum likelihood estimation, bias-variance tradeoff. Required for naive Bayes, Gaussian mixture models, and model evaluation.
- **Calculus**: Derivatives, partial derivatives, gradients, convexity. Required for gradient descent and for understanding why certain loss functions work.

Work through one textbook chapter at a time. Apply each concept immediately to a small code exercise.

### Phase 2: Supervised Learning – Foundational Methods

Learn these in order. Each builds on the previous.

1. **Linear regression** (ordinary least squares) – Understand the closed-form solution via normal equations and the gradient descent alternative. This is your baseline for regression tasks.
2. **Logistic regression** – The probability interpretation via sigmoid, cross-entropy loss, and why it is a linear classifier despite its name.
3. **k-Nearest Neighbors** – A nonparametric method. Understand the curse of dimensionality and why distance metrics matter.
4. **Decision trees** (ID3, C4.5, CART) – Splitting criteria (entropy, Gini index), overfitting via depth, and the difference between classification and regression trees.
5. **Support vector machines** – The margin concept, hard vs soft margin, kernel trick as a conceptual leap. Do not implement kernel SVM from scratch initially; understand what kernels do to feature spaces.
6. **Naive Bayes** – The conditional independence assumption, its surprising effectiveness, and its relationship to generative modeling.

For each method, implement a clean version from scratch on a small synthetic dataset. Then apply scikit-learn to a real dataset. Compare results.

### Phase 3: Unsupervised Learning – Foundational Methods

These require less prerequisite ordering. Learn them after Phase 2.

- **k-means clustering** – The expectation-maximization flavor, initialization sensitivity, and the Elbow method for choosing k.
- **Hierarchical clustering** – Agglomerative vs divisive, linkage criteria, and dendrogram interpretation.
- **Principal Component Analysis** (PCA) – Derivation via variance maximization or reconstruction error. This is your primary dimensionality reduction tool.
- **Gaussian mixture models** – Soft clustering and the full expectation-maximization algorithm.

### Phase 4: Model Evaluation and Selection

This phase is not optional. Without it, you cannot trust your own results.

- Train-test split and cross-validation (k-fold, stratified).
- Metrics for classification: accuracy, precision, recall, F1, ROC-AUC.
- Metrics for regression: MSE, MAE, R-squared.
- Overfitting detection via learning curves.
- Hyperparameter tuning: grid search vs random search.

### Phase 5: Core Practical Workflow

Before touching neural networks, complete three end-to-end projects of increasing complexity.

- **Project 1** – Binary classification on tabular data (e.g., Titanic, credit fraud). Practice: exploratory analysis, feature engineering, logistic regression vs decision tree, cross-validation.
- **Project 2** – Multiclass classification with feature extraction (e.g., handwritten digits from pixel values). Practice: PCA for visualization, SVM with different kernels, confusion matrix analysis.
- **Project 3** – Unsupervised learning as preprocessing for a supervised task (e.g., cluster labels as features, or PCA then classifier). Practice: pipeline construction in scikit-learn.

### Phase 6: Theory for Rigor (Optional but Recommended)

These topics separate practitioners who copy code from those who debug failures.

- Bias-variance decomposition – Derive it once on paper.
- No free lunch theorems – Understand what they actually claim (uniformity over all problems) and what they do not.
- VC dimension and PAC learning – At least the intuitive meaning: what makes a class of functions learnable and how much data suffices.
- Convexity in loss functions – Why convex losses guarantee global optima and why non-convex ones (e.g., in neural nets) do not.

### Implementation Rule

Use Python with NumPy for scratch implementations. Use scikit-learn for everything else. Do not use TensorFlow or PyTorch during this roadmap. They obscure the algorithmic structure you need to absorb.

### Timeline Estimate (Self-Paced)

- Phases 1–2: 6–8 weeks with daily practice
- Phase 3: 2–3 weeks
- Phase 4: 1 week integrated across all previous phases
- Phase 5: 4–6 weeks (one project every two weeks)
- Phase 6: ongoing, 1–2 papers or textbook chapters per week

Total: approximately 3–4 months to solid competence, not mastery. Mastery requires years of application to diverse domains.

### Recommended Single-Volume Reference

Hastie, Tibshirani, Friedman – *The Elements of Statistical Learning* (available free from the authors). Read chapters 1–9 in order. Skip the matrix algebra derivations on first pass but return to them.