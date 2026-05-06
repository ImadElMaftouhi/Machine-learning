# Linear Regression: A Rigorous Treatment

> **Written by:** Imad El Maftouhi

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Problem Formulation](#2-problem-formulation)
3. [The Ordinary Least Squares (OLS) Estimator](#3-the-ordinary-least-squares-ols-estimator)
4. [Geometric Interpretation](#4-geometric-interpretation)
5. [Statistical Properties of OLS](#5-statistical-properties-of-ols)
6. [The Gauss-Markov Theorem](#6-the-gauss-markov-theorem)
7. [Maximum Likelihood Estimation](#7-maximum-likelihood-estimation)
8. [Hypothesis Testing and Inference](#8-hypothesis-testing-and-inference)
9. [Goodness of Fit](#9-goodness-of-fit)
10. [Regularization: Ridge and Lasso](#10-regularization-ridge-and-lasso)
11. [Assumptions and Their Violations](#11-assumptions-and-their-violations)
12. [Polynomial and Nonlinear Extensions](#12-polynomial-and-nonlinear-extensions)
13. [Computational Considerations](#13-computational-considerations)
14. [Python Implementation](#14-python-implementation)

---

## 1. Introduction and Motivation

Linear regression is the foundational supervised learning algorithm for modeling the relationship between a continuous target variable and one or more explanatory variables. Despite its apparent simplicity, it underpins a vast portion of statistical theory, econometrics, and modern machine learning. Understanding it rigorously, including the mathematics of estimation, inference, and the geometry of projection, is an indispensable prerequisite for understanding more complex methods such as generalized linear models, kernel methods, and neural networks.

The core hypothesis is that the expected value of the response is a **linear function** of the inputs:

$$\mathbb{E}[y \mid \mathbf{x}] = f(\mathbf{x}) = \mathbf{x}^\top \boldsymbol{\beta}$$

where linearity is with respect to the **parameters** $\boldsymbol{\beta}$, not necessarily with respect to the features $\mathbf{x}$ (which may be nonlinearly transformed).

$\mathbf{x}^\top \boldsymbol{\beta}$, $\mathbf{x} \in \mathbb{R}^{p+1}$ is  a **column vector** (the convention throughout the document), so a single observation's feature vector is:

$$\mathbf{x} = \begin{pmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_p \end{pmatrix} \in \mathbb{R}^{p+1}$$

and $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$ is also a column vector. The inner product $\mathbf{x}^\top \boldsymbol{\beta}$ is then a row vector times a column vector, yielding a scalar, which is the correct type for $\mathbb{E}[y \mid \mathbf{x}]$.

Writing it as $\boldsymbol{\beta}^\top \mathbf{x}$ would be equally valid and perhaps more common in machine learning literature. Both express the same scalar dot product.

---

## 2. Problem Formulation

### 2.1 Simple Linear Regression

Given $n$ observations $\{(x_i, y_i)\}_{i=1}^n$, the simple linear regression model is:

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad i = 1, \ldots, n$$

where $\beta_0$ is the intercept, $\beta_1$ is the slope, and $\varepsilon_i$ is a random error term.

### 2.2 Multiple Linear Regression

With $p$ predictors, the model generalizes to:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i$$

In matrix-vector notation, collecting all $n$ observations:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

where:

- $\mathbf{y} \in \mathbb{R}^n$ is the response vector
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ is the **design matrix**, with a leading column of ones for the intercept
- $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$ is the parameter vector
- $\boldsymbol{\varepsilon} \in \mathbb{R}^n$ is the error vector

Explicitly:

$$\begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}
=
\begin{pmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{pmatrix}
\begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{pmatrix}
+
\begin{pmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{pmatrix}$$

---

## 3. The Ordinary Least Squares (OLS) Estimator

### 3.1 Objective Function

OLS minimizes the **Residual Sum of Squares (RSS)**:

$$\text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Expanding:

$$\text{RSS}(\boldsymbol{\beta}) = \mathbf{y}^\top\mathbf{y} - 2\boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{y} + \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}$$

### 3.2 Derivation of the Normal Equations

Taking the gradient with respect to $\boldsymbol{\beta}$ and setting it to zero:

$$\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$$

This yields the **Normal Equations**:

$$\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}$$

Provided $\mathbf{X}^\top\mathbf{X}$ is invertible (i.e., $\mathbf{X}$ has full column rank, $\text{rank}(\mathbf{X}) = p+1$), the unique closed-form solution is:

$$\boxed{\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}}$$

To confirm this is a minimum (not a saddle or maximum), examine the Hessian:

$$\mathbf{H} = \frac{\partial^2 \text{RSS}}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^\top} = 2\mathbf{X}^\top\mathbf{X}$$

Since $\mathbf{X}^\top\mathbf{X}$ is positive semi-definite (positive definite when $\mathbf{X}$ has full rank), the RSS is strictly convex and the solution is a global minimum.

### 3.3 Fitted Values and Residuals

The vector of **fitted values** is:

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \mathbf{H}\mathbf{y}$$

where $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \in \mathbb{R}^{n \times n}$ is the **hat matrix** (or projection matrix).

The residuals are:

$$\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$$

Key properties of $\mathbf{H}$ and $(\mathbf{I} - \mathbf{H})$:

| Property | Hat Matrix $\mathbf{H}$ | Residual Maker $\mathbf{M} = \mathbf{I} - \mathbf{H}$ |
|----------|------------------------|-------------------------------------------------------|
| Symmetry | $\mathbf{H}^\top = \mathbf{H}$ | $\mathbf{M}^\top = \mathbf{M}$ |
| Idempotency | $\mathbf{H}^2 = \mathbf{H}$ | $\mathbf{M}^2 = \mathbf{M}$ |
| Rank | $\text{rank}(\mathbf{H}) = p+1$ | $\text{rank}(\mathbf{M}) = n - p - 1$ |
| Projection onto | Column space of $\mathbf{X}$ | Orthogonal complement of col($\mathbf{X}$) |

---

## 4. Geometric Interpretation

The OLS solution is a **projection**. The column space of $\mathbf{X}$, denoted $\text{col}(\mathbf{X})$, is a $(p+1)$-dimensional subspace of $\mathbb{R}^n$. Since $\mathbf{y}$ generally does not lie in this subspace, we seek the point in $\text{col}(\mathbf{X})$ closest (in Euclidean distance) to $\mathbf{y}$.

That closest point is the orthogonal projection $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$. The residual vector $\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every column of $\mathbf{X}$:

$$\mathbf{X}^\top \hat{\boldsymbol{\varepsilon}} = \mathbf{0}$$

This orthogonality condition is exactly the Normal Equations. Geometrically, it means the residuals are perpendicular to the predictor space, which is the fundamental insight of OLS.

---

## 5. Statistical Properties of OLS

The following properties hold under the classical assumptions: (A1) linearity, (A2) strict exogeneity $\mathbb{E}[\boldsymbol{\varepsilon} \mid \mathbf{X}] = \mathbf{0}$, (A3) spherical errors $\text{Var}[\boldsymbol{\varepsilon} \mid \mathbf{X}] = \sigma^2 \mathbf{I}$, (A4) full rank of $\mathbf{X}$.

### 5.1 Unbiasedness

$$\mathbb{E}[\hat{\boldsymbol{\beta}} \mid \mathbf{X}] = \mathbb{E}[(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} \mid \mathbf{X}]$$

$$= (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \mathbb{E}[\mathbf{y} \mid \mathbf{X}]$$

$$= (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \mathbf{X}\boldsymbol{\beta} = \boldsymbol{\beta}$$

Hence $\hat{\boldsymbol{\beta}}$ is unbiased for $\boldsymbol{\beta}$.

### 5.2 Variance-Covariance Matrix

$$\text{Var}[\hat{\boldsymbol{\beta}} \mid \mathbf{X}] = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \cdot \sigma^2\mathbf{I} \cdot \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}$$

$$= \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}$$

The standard error of the $j$-th coefficient is:

$$\text{SE}(\hat{\beta}_j) = \hat{\sigma}\sqrt{[(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj}}$$

where $\hat{\sigma}^2$ is the **unbiased estimator of the error variance**:

$$\hat{\sigma}^2 = \frac{\hat{\boldsymbol{\varepsilon}}^\top\hat{\boldsymbol{\varepsilon}}}{n - p - 1} = \frac{\text{RSS}}{n - p - 1}$$

The denominator $n - p - 1$ reflects the degrees of freedom consumed by estimating $p+1$ parameters.

### 5.3 Consistency

As $n \to \infty$, under mild regularity conditions:

$$\hat{\boldsymbol{\beta}} \xrightarrow{p} \boldsymbol{\beta}$$

This follows from the Law of Large Numbers applied to $\frac{1}{n}\mathbf{X}^\top\mathbf{X} \to \boldsymbol{\Sigma}_{XX}$ and $\frac{1}{n}\mathbf{X}^\top\mathbf{y} \to \boldsymbol{\Sigma}_{XX}\boldsymbol{\beta}$.

### 5.4 Asymptotic Normality

By the Central Limit Theorem:

$$\sqrt{n}(\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}) \xrightarrow{d} \mathcal{N}\left(\mathbf{0},\ \sigma^2 \boldsymbol{\Sigma}_{XX}^{-1}\right)$$

Under the additional assumption that $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, the distribution is **exact** for any $n$:

$$\hat{\boldsymbol{\beta}} \mid \mathbf{X} \sim \mathcal{N}\left(\boldsymbol{\beta},\ \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}\right)$$

---

## 6. The Gauss-Markov Theorem

**Theorem.** Under assumptions (A1)-(A4), among all **linear unbiased estimators** of $\boldsymbol{\beta}$, the OLS estimator $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ has the **smallest variance** (is BLUE: Best Linear Unbiased Estimator).

**Proof sketch.** Let $\tilde{\boldsymbol{\beta}} = \mathbf{C}\mathbf{y}$ be any other linear unbiased estimator, where $\mathbf{C} \neq (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$. Define $\mathbf{D} = \mathbf{C} - (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$. Unbiasedness requires $\mathbf{C}\mathbf{X} = \mathbf{I}$, which implies $\mathbf{D}\mathbf{X} = \mathbf{0}$.

Then:

$$\text{Var}[\tilde{\boldsymbol{\beta}}] = \sigma^2 \mathbf{C}\mathbf{C}^\top = \sigma^2\left[(\mathbf{X}^\top\mathbf{X})^{-1} + \mathbf{D}\mathbf{D}^\top\right]$$

Since $\mathbf{D}\mathbf{D}^\top$ is positive semi-definite:

$$\text{Var}[\tilde{\boldsymbol{\beta}}] \succeq \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1} = \text{Var}[\hat{\boldsymbol{\beta}}_{\text{OLS}}]$$

This is the Loewner partial order on positive semi-definite matrices, meaning the difference is PSD.

---

## 7. Maximum Likelihood Estimation

Assume $\varepsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$. Then $y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{x}_i^\top\boldsymbol{\beta}, \sigma^2)$ and the log-likelihood is:

$$\ell(\boldsymbol{\beta}, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$$

Maximizing over $\boldsymbol{\beta}$ is equivalent to minimizing $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$. Therefore:

$$\hat{\boldsymbol{\beta}}_{\text{MLE}} = \hat{\boldsymbol{\beta}}_{\text{OLS}}$$

Maximizing over $\sigma^2$ gives:

$$\hat{\sigma}^2_{\text{MLE}} = \frac{\text{RSS}}{n}$$

Note that the MLE of $\sigma^2$ is **biased** (divides by $n$ rather than $n-p-1$). The unbiased correction substitutes $n - p - 1$ in the denominator.

---

## 8. Hypothesis Testing and Inference

### 8.1 $t$-Test for Individual Coefficients

To test $H_0: \beta_j = 0$ against $H_1: \beta_j \neq 0$, use the test statistic:

$$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} = \frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{[(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj}}}$$

Under $H_0$, $t_j \sim t_{n-p-1}$ (Student's $t$ distribution with $n-p-1$ degrees of freedom). The null is rejected at level $\alpha$ if $|t_j| > t_{n-p-1, 1-\alpha/2}$.

### 8.2 Confidence Intervals

A $(1-\alpha)$ confidence interval for $\beta_j$ is:

$$\hat{\beta}_j \pm t_{n-p-1,\, 1-\alpha/2} \cdot \text{SE}(\hat{\beta}_j)$$

### 8.3 $F$-Test for Overall Significance

To test the joint hypothesis $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$:

$$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n-p-1)} = \frac{\text{MSR}}{\text{MSE}}$$

Under $H_0$, $F \sim F_{p,\, n-p-1}$. This tests whether the model as a whole explains significant variance in $\mathbf{y}$.

### 8.4 Testing Linear Restrictions

For a general linear hypothesis $H_0: \mathbf{R}\boldsymbol{\beta} = \mathbf{r}$ where $\mathbf{R} \in \mathbb{R}^{q \times (p+1)}$ has rank $q$:

$$F = \frac{(\mathbf{R}\hat{\boldsymbol{\beta}} - \mathbf{r})^\top \left[\mathbf{R}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{R}^\top\right]^{-1} (\mathbf{R}\hat{\boldsymbol{\beta}} - \mathbf{r})}{q\hat{\sigma}^2} \sim F_{q,\, n-p-1}$$

---

## 9. Goodness of Fit

### 9.1 Decomposition of Variance

The fundamental ANOVA decomposition is:

$$\underbrace{\sum_{i=1}^n (y_i - \bar{y})^2}_{\text{TSS}} = \underbrace{\sum_{i=1}^n (\hat{y}_i - \bar{y})^2}_{\text{ESS}} + \underbrace{\sum_{i=1}^n (y_i - \hat{y}_i)^2}_{\text{RSS}}$$

That is, $\text{TSS} = \text{ESS} + \text{RSS}$. (This requires an intercept in the model.)

### 9.2 Coefficient of Determination $R^2$

$$R^2 = \frac{\text{ESS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}} \in [0, 1]$$

$R^2$ measures the proportion of total variance explained by the model. It is also the squared Pearson correlation between $\mathbf{y}$ and $\hat{\mathbf{y}}$.

**Limitation:** $R^2$ is non-decreasing in the number of predictors, even if added predictors are irrelevant. This leads to overfitting in model selection.

### 9.3 Adjusted $R^2$

To penalize model complexity:

$$\bar{R}^2 = 1 - \frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)} = 1 - (1 - R^2)\frac{n-1}{n-p-1}$$

Unlike $R^2$, $\bar{R}^2$ can decrease when a predictor adds insufficient explanatory power.

### 9.4 Information Criteria

For model selection, two principled criteria are widely used:

$$\text{AIC} = n\log\left(\frac{\text{RSS}}{n}\right) + 2(p+2)$$

$$\text{BIC} = n\log\left(\frac{\text{RSS}}{n}\right) + (p+2)\log(n)$$

BIC applies a heavier complexity penalty than AIC and is consistent (selects the true model as $n \to \infty$ under the true model).

---

## 10. Regularization: Ridge and Lasso

When $p$ is large relative to $n$ or when $\mathbf{X}^\top\mathbf{X}$ is near-singular (multicollinearity), OLS is numerically unstable and has high variance. Regularization introduces a penalty on the parameters.

### 10.1 Ridge Regression ($\ell_2$ Penalty)

$$\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_2^2 \right\}$$

The closed-form solution is:

$$\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

The matrix $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ is always invertible for $\lambda > 0$, even if $\mathbf{X}^\top\mathbf{X}$ is singular.

**Bias-Variance Tradeoff.** Ridge introduces bias but can dramatically reduce variance:

$$\text{Bias}[\hat{\boldsymbol{\beta}}_{\text{ridge}}] = -\lambda(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}$$

$$\text{Var}[\hat{\boldsymbol{\beta}}_{\text{ridge}}] = \sigma^2(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}$$

Using the SVD $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$, the ridge estimator shrinks each singular direction by a factor $d_j^2/(d_j^2 + \lambda)$.

### 10.2 Lasso ($\ell_1$ Penalty)

$$\hat{\boldsymbol{\beta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_1 \right\}$$

The $\ell_1$ penalty is non-differentiable at zero, producing **exact sparsity**: many coefficients are driven to exactly zero. This makes Lasso perform implicit variable selection.

There is no closed-form solution in general; the problem is solved via coordinate descent or the LARS algorithm. The subgradient condition yields the soft-thresholding operator:

$$\hat{\beta}_j^{\text{lasso}} = \text{sign}(\hat{\beta}_j^{\text{OLS}}) \cdot \max\left(|\hat{\beta}_j^{\text{OLS}}| - \frac{\lambda}{2}, 0\right) \quad \text{(orthogonal design only)}$$

### 10.3 Elastic Net

Combines both penalties:

$$\hat{\boldsymbol{\beta}}_{\text{EN}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1\|\boldsymbol{\beta}\|_1 + \lambda_2\|\boldsymbol{\beta}\|_2^2 \right\}$$

This achieves sparsity while handling correlated predictors better than Lasso alone.

---

## 11. Assumptions and Their Violations

The reliability of OLS inference depends critically on the classical assumptions. Below is a rigorous treatment of each.

### 11.1 Linearity

The model assumes $\mathbb{E}[y \mid \mathbf{x}] = \mathbf{x}^\top\boldsymbol{\beta}$. **Violation:** misspecification bias. Remedies include adding polynomial terms, interaction terms, or switching to a nonparametric model.

### 11.2 Exogeneity ($\mathbb{E}[\varepsilon \mid \mathbf{X}] = \mathbf{0}$)

This is violated by **omitted variable bias** (a relevant variable correlated with both $\mathbf{x}$ and $\varepsilon$ is excluded), **simultaneity**, or **measurement error** in regressors. In such cases, OLS is biased and inconsistent. Remedies include instrumental variable (IV) estimation.

### 11.3 Homoskedasticity ($\text{Var}[\varepsilon_i \mid \mathbf{x}_i] = \sigma^2$)

**Heteroskedasticity** occurs when variance is non-constant. OLS remains unbiased but is no longer efficient, and standard errors are incorrect. Remedies: use **Heteroskedasticity-Consistent (HC) standard errors** (White's sandwich estimator) or **Weighted Least Squares (WLS)**:

$$\hat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}\mathbf{y}$$

where $\mathbf{W} = \text{diag}(w_1, \ldots, w_n)$ with $w_i = 1/\text{Var}[\varepsilon_i \mid \mathbf{x}_i]$.

### 11.4 No Autocorrelation ($\text{Cov}[\varepsilon_i, \varepsilon_j \mid \mathbf{X}] = 0$ for $i \neq j$)

Time-series data frequently exhibit **serial correlation**. The Durbin-Watson test detects first-order autocorrelation. Remedies include Generalized Least Squares (GLS) or Newey-West standard errors.

### 11.5 No Perfect Multicollinearity

$\mathbf{X}$ must have full column rank. **Perfect multicollinearity** (an exact linear relationship among predictors) makes $\mathbf{X}^\top\mathbf{X}$ singular and OLS undefined. **Near-multicollinearity** inflates standard errors. The **Variance Inflation Factor** for predictor $j$ is:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the $R^2$ from regressing $x_j$ on all other predictors. A $\text{VIF}_j > 10$ is a common warning threshold.

### 11.6 Normality of Errors

Required for exact $t$ and $F$ distributions. With large $n$, the CLT makes inference approximately valid without normality. Assessed via Q-Q plots and the Shapiro-Wilk test.

---

## 12. Polynomial and Nonlinear Extensions

### 12.1 Polynomial Regression

Remain within the linear regression framework by augmenting the feature space with polynomial terms:

$$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_d x_i^d + \varepsilon_i$$

This is still a linear model in $\boldsymbol{\beta}$, so all OLS theory applies. The design matrix simply includes powers of $x$ as columns.

### 12.2 Basis Expansion

More generally, define $\phi: \mathbb{R}^p \to \mathbb{R}^m$ as a vector of basis functions (splines, Fourier bases, RBF kernels). The model becomes:

$$y_i = \boldsymbol{\beta}^\top \boldsymbol{\phi}(\mathbf{x}_i) + \varepsilon_i$$

This is linear in $\boldsymbol{\beta}$ and the OLS solution is:

$$\hat{\boldsymbol{\beta}} = (\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\mathbf{y}$$

where $\boldsymbol{\Phi}_{ij} = \phi_j(\mathbf{x}_i)$ is the transformed design matrix.

### 12.3 The Kernel Trick

For $m \gg n$, it is computationally preferable to work in the dual space. Using the kernel trick with $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{x}')$, the predictions become:

$$\hat{\mathbf{y}} = \mathbf{K}(\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{y}$$

where $\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is the $n \times n$ kernel matrix (Kernel Ridge Regression).

---

## 13. Computational Considerations

### 13.1 Direct Inversion vs. Linear Solve

Never compute $(\mathbf{X}^\top\mathbf{X})^{-1}$ explicitly in practice. Instead, solve the normal equations as a linear system:

$$(\mathbf{X}^\top\mathbf{X})\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$$

using Cholesky decomposition (when $\mathbf{X}^\top\mathbf{X}$ is SPD), which requires $O((p+1)^3)$ flops.

### 13.2 QR Decomposition

The numerically preferred approach uses the QR decomposition of $\mathbf{X} = \mathbf{Q}\mathbf{R}$, where $\mathbf{Q} \in \mathbb{R}^{n \times (p+1)}$ has orthonormal columns and $\mathbf{R} \in \mathbb{R}^{(p+1) \times (p+1)}$ is upper triangular. Substituting:

$$\hat{\boldsymbol{\beta}} = (\mathbf{R}^\top\mathbf{Q}^\top\mathbf{Q}\mathbf{R})^{-1}\mathbf{R}^\top\mathbf{Q}^\top\mathbf{y} = \mathbf{R}^{-1}\mathbf{Q}^\top\mathbf{y}$$

This is solved by back-substitution. QR avoids squaring the condition number of $\mathbf{X}$ (as occurs when forming $\mathbf{X}^\top\mathbf{X}$), and is numerically stable with $O(n(p+1)^2)$ complexity.

### 13.3 SVD Decomposition

The most robust approach uses the SVD $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$:

$$\hat{\boldsymbol{\beta}} = \mathbf{V}\mathbf{D}^{-1}\mathbf{U}^\top\mathbf{y}$$

When $\mathbf{X}$ is rank-deficient, the Moore-Penrose pseudoinverse $\mathbf{X}^+ = \mathbf{V}\mathbf{D}^+\mathbf{U}^\top$ (zeroing reciprocals of zero singular values) yields the minimum-norm OLS solution:

$$\hat{\boldsymbol{\beta}}_{\text{min-norm}} = \mathbf{X}^+\mathbf{y}$$

### 13.4 Stochastic Gradient Descent

For large-scale settings ($n$ and $p$ both large), gradient descent is used. The gradient of RSS is:

$$\nabla_{\boldsymbol{\beta}} \text{RSS} = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Mini-batch SGD uses a random subset $\mathcal{B} \subset [n]$ at each step:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + \frac{2\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (y_i - \mathbf{x}_i^\top\boldsymbol{\beta}^{(t)})\mathbf{x}_i$$

---

## 14. Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.linalg as la


# ─── Data generation ────────────────────────────────────────────────────────

np.random.seed(42)
X, y, true_coef = make_regression(
    n_samples=200, n_features=10, n_informative=6,
    noise=15.0, coef=True, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ─── OLS via Normal Equations ────────────────────────────────────────────────

def ols_normal_equations(X, y):
    """Solve OLS via the Normal Equations using Cholesky decomposition."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    A = X_aug.T @ X_aug
    b = X_aug.T @ y
    beta = la.solve(A, b, assume_a='pos')  # Cholesky solve (SPD matrix)
    return beta


# ─── OLS via QR Decomposition ────────────────────────────────────────────────

def ols_qr(X, y):
    """Solve OLS via thin QR decomposition (numerically stable)."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    Q, R = la.qr(X_aug, mode='economic')
    beta = la.solve_triangular(R, Q.T @ y)
    return beta


# ─── Inference: Standard Errors, t-stats, p-values ──────────────────────────

def ols_inference(X, y):
    """Full OLS inference: coefficients, SEs, t-stats, confidence intervals."""
    from scipy import stats

    n, p = X.shape
    X_aug = np.column_stack([np.ones(n), X])
    k = X_aug.shape[1]

    beta = ols_qr(X, y)
    y_hat = X_aug @ beta
    residuals = y - y_hat
    RSS = np.dot(residuals, residuals)

    sigma2_hat = RSS / (n - k)                   # unbiased estimator
    XtX_inv = la.inv(X_aug.T @ X_aug)
    cov_beta = sigma2_hat * XtX_inv
    se = np.sqrt(np.diag(cov_beta))

    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    ci_lower = beta - stats.t.ppf(0.975, df=n - k) * se
    ci_upper = beta + stats.t.ppf(0.975, df=n - k) * se

    y_bar = np.mean(y)
    TSS = np.sum((y - y_bar) ** 2)
    ESS = np.sum((y_hat - y_bar) ** 2)
    R2 = ESS / TSS
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - k)

    F_stat = (ESS / (k - 1)) / (RSS / (n - k))
    F_pval = 1 - stats.f.cdf(F_stat, dfn=k - 1, dfd=n - k)

    return {
        "beta": beta, "se": se,
        "t_stats": t_stats, "p_values": p_values,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "R2": R2, "R2_adj": R2_adj,
        "sigma2": sigma2_hat,
        "F_stat": F_stat, "F_pval": F_pval,
        "RSS": RSS, "TSS": TSS,
    }


# ─── Regularized Regression ──────────────────────────────────────────────────

def ridge_closed_form(X, y, lam):
    """Ridge regression via closed-form solution."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    k = X_aug.shape[1]
    A = X_aug.T @ X_aug + lam * np.eye(k)
    return la.solve(A, X_aug.T @ y)


# ─── Run and Report ──────────────────────────────────────────────────────────

results = ols_inference(X_train, y_train)

print("=" * 55)
print(f"{'OLS Inference Summary':^55}")
print("=" * 55)
print(f"  R²         = {results['R2']:.4f}")
print(f"  Adj. R²    = {results['R2_adj']:.4f}")
print(f"  sigma²     = {results['sigma2']:.4f}")
print(f"  F-stat     = {results['F_stat']:.2f}  (p = {results['F_pval']:.4e})")
print("-" * 55)
print(f"{'Coeff':>10}  {'SE':>8}  {'t-stat':>8}  {'p-val':>10}  {'95% CI'}")
print("-" * 55)
for i, (b, s, t, p, lo, hi) in enumerate(zip(
    results["beta"], results["se"], results["t_stats"],
    results["p_values"], results["ci_lower"], results["ci_upper"]
)):
    name = "Intercept" if i == 0 else f"x{i}"
    sig = "*" if p < 0.05 else ""
    print(f"{name:>10}  {b:>8.3f}  {s:>8.3f}  {t:>8.3f}  {p:>10.4f}  [{lo:.3f}, {hi:.3f}] {sig}")
print("=" * 55)

# Generalization performance
beta_ols = ols_qr(X_train, y_train)
X_test_aug = np.column_stack([np.ones(len(X_test)), X_test])
y_pred_ols = X_test_aug @ beta_ols

print(f"\nTest MSE (OLS): {mean_squared_error(y_test, y_pred_ols):.4f}")
print(f"Test R²  (OLS): {r2_score(y_test, y_pred_ols):.4f}")

# Compare with sklearn (sanity check)
lr = LinearRegression().fit(X_train, y_train)
print(f"Test R²  (sklearn): {lr.score(X_test, y_test):.4f}")
```

---

## Summary of Key Results

| Concept | Formula |
|---|---|
| OLS Estimator | $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| Hat Matrix | $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ |
| Error Variance | $\hat{\sigma}^2 = \text{RSS}/(n-p-1)$ |
| Coefficient Variance | $\text{Var}[\hat{\boldsymbol{\beta}}] = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$ |
| $R^2$ | $1 - \text{RSS}/\text{TSS}$ |
| Ridge | $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| $t$-statistic | $\hat{\beta}_j / \text{SE}(\hat{\beta}_j)$ |
| $F$-statistic | $(\text{ESS}/p) / (\text{RSS}/(n-p-1))$ |

---

*Document prepared for graduate-level study. All derivations assume familiarity with linear algebra, multivariate calculus, and probability theory at the undergraduate level.*