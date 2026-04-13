# Regression

Regression is the supervised learning task where the goal is to predict a continuous output value from input features. The model learns a mapping from an input vector to a real number — for instance, predicting house prices from square footage and location, or forecasting temperature from atmospheric measurements.

The defining characteristic of regression is its output space: real-valued and unbounded (or bounded by domain constraints, not by the model itself). This distinguishes it from classification, where the output is drawn from a finite discrete set.

# Why Start Here

Regression is the right entry point into supervised learning because its loss functions are differentiable and geometrically intuitive, its closed-form solutions reveal the linear algebra underlying many other methods, and its failure modes — underfitting and overfitting — are easy to visualize through residual plots and learning curves.

Understanding linear regression thoroughly — not just calling `sklearn.LinearRegression` — is a prerequisite for understanding regularization, gradient descent, and the bias-variance tradeoff, all of which carry forward into every other method in this module.

# Methods

## Linear Regression (Ordinary Least Squares)

Linear regression models the output as a weighted sum of input features plus a bias term. Training finds the weights that minimize the sum of squared residuals between predictions and targets.

There are two routes to the solution. The **closed-form solution** via the normal equations computes the exact optimum in one step using matrix algebra. It is exact but scales poorly to large feature sets (matrix inversion is O(n³)). **Gradient descent** iteratively updates weights in the direction of the negative gradient of the loss. It is approximate but scales to any dataset size and generalizes directly to neural networks.

Implement both and compare results. They should agree on small datasets.

## Ridge Regression (L2 Regularization)

Ridge regression adds a penalty proportional to the sum of squared weights to the ordinary least squares loss. This shrinks weights toward zero, reducing variance at the cost of a small increase in bias — the classic bias-variance tradeoff in action.

Ridge is the first introduction to **regularization**: the idea that constraining model complexity during training improves generalization to unseen data. The regularization strength λ is a hyperparameter; too small and it has no effect, too large and the model underfits.

## Polynomial Regression

Polynomial regression fits a nonlinear relationship between input and output by constructing polynomial features from the original inputs (x → x, x², x³, ...) and then applying linear regression on the expanded feature set. The model is still linear in the parameters — only the features are nonlinear.

This is a useful demonstration that "linear" in linear regression refers to linearity in the parameters, not in the input. It also vividly demonstrates overfitting: a high-degree polynomial can fit any training set perfectly while generalizing poorly.

# Files in This Folder

| File | Contents |
|---|---|
| `linear_regression.py` | OLS via normal equations and gradient descent |
| `ridge_regression.py` | Ridge regression with tunable λ |
| `polynomial_regression.py` | Polynomial feature construction + linear regression |
| `regression_scratch_vs_sklearn.ipynb` | Validation of scratch implementations against scikit-learn |
| `regression_real_world.ipynb` | Applied regression on a real dataset with EDA and residual analysis |

# Evaluation Metrics

- **Mean Squared Error (MSE)**: penalizes large errors more than small ones due to squaring. The same metric minimized during training.
- **Mean Absolute Error (MAE)**: more robust to outliers than MSE; measures average absolute deviation.
- **R² (coefficient of determination)**: proportion of variance in the target explained by the model. A value of 1.0 means perfect fit; 0.0 means the model does no better than predicting the mean.

# What to Observe

- Compare the closed-form and gradient descent solutions on the same dataset. Verify they converge to the same weights.
- Plot residuals (predicted minus actual) against predicted values. A good model has residuals scattered randomly around zero with no pattern.
- Increase polynomial degree and watch training error fall while validation error rises — this is overfitting made visible.
- Vary λ in ridge regression and plot the resulting weight norms. Observe how regularization shrinks weights.
