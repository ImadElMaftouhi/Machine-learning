# Dimensionality Reduction

Dimensionality reduction transforms a dataset from a high-dimensional space into a lower-dimensional one while preserving as much relevant structure as possible. The goal is to discard dimensions that carry little information — either redundant features, noise, or directions of low variance — and retain the directions that matter most.

High-dimensional data is pervasive in practice: image pixels, text feature vectors, genomic measurements. Working in very high dimensions creates problems: distance metrics become uninformative (the curse of dimensionality), visualization is impossible, and many algorithms become slow or statistically unreliable. Dimensionality reduction addresses all three.

# Two Perspectives on Reduction

There are two complementary ways to motivate dimensionality reduction.

**Unsupervised**: find directions in the original space that capture the most variance in the data, without reference to any labels. Principal Component Analysis (PCA) is the canonical method. It is useful for exploratory analysis, visualization, denoising, and as a preprocessing step before any downstream task.

**Supervised**: find directions that best separate the classes in the data. Linear Discriminant Analysis (LDA) is the canonical method. It uses label information to find a projection that maximizes between-class scatter relative to within-class scatter. LDA is both a dimensionality reduction method and a classifier.

# Methods

## Principal Component Analysis (PCA)

PCA finds a set of orthogonal directions — principal components — ordered by the amount of variance they explain. Projecting the data onto the top k components gives the best k-dimensional linear approximation of the original data in terms of reconstruction error (equivalently, in terms of retained variance).

The derivation has two equivalent formulations: maximize the variance of the projected data, or minimize the reconstruction error when projecting and then projecting back. Both lead to the same solution: the principal components are the eigenvectors of the data's covariance matrix, ordered by descending eigenvalue.

In practice, PCA is computed via **Singular Value Decomposition (SVD)** rather than by explicitly constructing the covariance matrix, which is more numerically stable and efficient.

Key decisions: how many components to retain. Plot the **explained variance ratio** (each eigenvalue divided by the sum of all eigenvalues) against component index. Choose k at the "elbow" of the curve, or at a threshold (e.g., retain enough components to explain 95% of total variance).

## Linear Discriminant Analysis (LDA)

LDA finds a projection that maximizes the ratio of between-class scatter to within-class scatter. Intuitively, it finds directions along which the classes are most separated, while keeping each class's internal spread small.

LDA differs from PCA in a critical way: it uses class labels. This makes it a supervised dimensionality reduction method. The maximum number of discriminant components is min(n_classes − 1, n_features), so LDA is most useful for multi-class problems.

LDA can also be used directly as a classifier: project the data and assign each point to the class whose projected mean it is closest to (under a Gaussian assumption). This is equivalent to a Bayes-optimal classifier when the classes have equal covariance matrices.

# Files in This Folder

| File | Contents |
|---|---|
| `pca.py` | PCA via SVD; fit, transform, inverse_transform, explained variance ratio |
| `lda.py` | LDA via scatter matrix generalized eigenvalue problem |
| `pca_exploration.ipynb` | PCA on a high-dimensional dataset; explained variance plots, component visualization, reconstruction |
| `pca_then_classify.ipynb` | PCA as preprocessing: vary k components, measure downstream classification accuracy |
| `lda_vs_pca.ipynb` | Direct comparison of supervised vs. unsupervised reduction on the same dataset |

# Evaluation

Dimensionality reduction is a preprocessing step, so it is usually evaluated indirectly:

- **Explained variance ratio**: for PCA, how much of the total variance is retained by k components. Plot the cumulative curve and look for an elbow.
- **Reconstruction error**: project to k dimensions and project back. Measure MSE between original and reconstructed data.
- **Downstream task performance**: apply reduction then a classifier or regressor. Plot accuracy (or another metric) vs. number of retained components. There is usually a sweet spot.
- **Visualization**: project to 2 or 3 components and plot. If the data has meaningful structure, it should be visible.

# What to Observe

- Apply PCA to a dataset of digit images (e.g., MNIST subset). Visualize the first few principal components as images — they should look like abstract digit-like patterns ("eigendigits"). Reconstruct a digit from k components and watch quality improve as k increases.
- Compare PCA and LDA on a multiclass dataset (e.g., Iris). Project to 2 dimensions with each method and plot. LDA should produce cleaner class separation because it uses label information.
- Run a classifier (e.g., logistic regression) on the raw features vs. PCA-reduced features at various k values. Plot the accuracy vs. k curve. Observe that a small number of components often recovers most of the predictive performance.
- Check the condition of the within-class scatter matrix before running LDA. If it is near-singular (common when features outnumber samples), regularization is needed — this is the connection to Regularized LDA.
