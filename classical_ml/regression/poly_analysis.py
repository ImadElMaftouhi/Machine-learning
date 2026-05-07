# regression/poly_analysis.py
#
# Performance & behavior study of the custom PolynomialRegression.
# Unique to polynomial regression vs. linear/ridge:
#   - bias-variance is controlled directly by the polynomial degree,
#   - high-degree fits are usable only with regularization,
#   - the fitted curve is intrinsically interpretable in 1D.

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from polynomial_regression import PolynomialRegression

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from time import perf_counter


ALL_METHODS = ["cholesky", "qr", "svd", "gradient_descent"]

# Default solver for the bias-variance sweeps: SVD is the most numerically
# robust at high polynomial degrees (cf. linear_regression.md §13.3).
DEFAULT_METHOD = "svd"


def _fit_and_score(model, X_train, y_train, X_test, y_test):
    t0 = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - t0

    y_pred_te = model.predict(X_test)
    y_pred_tr = model.predict(X_train)
    mse_te = mean_squared_error(y_test, y_pred_te)
    return {
        "fit_time": fit_time,
        "train_r2": r2_score(y_train, y_pred_tr),
        "test_r2":  r2_score(y_test,  y_pred_te),
        "mse":  mse_te,
        "rmse": np.sqrt(mse_te),
        "mae":  mean_absolute_error(y_test, y_pred_te),
    }


def _make_data_1d(n_samples=200, noise=0.4, x_range=(-2.0, 2.0), random_state=42):
    """Smooth non-linear ground truth: y = sin(1.5x) + 0.5x + Gaussian noise."""
    rng = np.random.default_rng(random_state)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y_clean = np.sin(1.5 * x) + 0.5 * x
    y = y_clean + rng.normal(0.0, noise, n_samples)
    X = x.reshape(-1, 1)
    return train_test_split(X, y, test_size=0.25, random_state=random_state)


# ---------- Experiment 1: degree sweep (the canonical bias-variance demo) ----------
def sweep_degree(method=DEFAULT_METHOD, degree_grid=range(1, 13),
                 n_samples=150, noise=0.4):
    X_tr, X_te, y_tr, y_te = _make_data_1d(n_samples=n_samples, noise=noise)
    rows = []
    for d in degree_grid:
        model = PolynomialRegression(degree=d, method=method, penalty=0.0)
        scores = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
        rows.append({"degree": d, **scores})
    return pd.DataFrame(rows)


# ---------- Experiment 2: learning curve (size of training set) ----------
def sweep_n_samples(method=DEFAULT_METHOD, degree=8, noise=0.4,
                    sizes=(20, 40, 80, 160, 320, 640, 1280, 2560)):
    rows = []
    for n in sizes:
        X_tr, X_te, y_tr, y_te = _make_data_1d(n_samples=n, noise=noise)
        model = PolynomialRegression(degree=degree, method=method, penalty=0.0)
        scores = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
        rows.append({"n_samples": n, **scores})
    return pd.DataFrame(rows)


# ---------- Experiment 3: regularization at high degree ----------
def sweep_penalty(method=DEFAULT_METHOD, degree=12,
                  lambdas=np.logspace(-5, 4, 30),
                  n_samples=150, noise=0.4):
    X_tr, X_te, y_tr, y_te = _make_data_1d(n_samples=n_samples, noise=noise)
    rows = []
    for lam in lambdas:
        model = PolynomialRegression(degree=degree, method=method, penalty=lam)
        scores = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
        rows.append({"lambda": lam, **scores})
    return pd.DataFrame(rows)


# ---------- Experiment 4: solver method comparison at moderate degree ----------
def compare_methods(degree=6, n_samples=2000, noise=0.4, n_repeats=5):
    X_tr, X_te, y_tr, y_te = _make_data_1d(n_samples=n_samples, noise=noise)
    rows = []
    for method in ALL_METHODS:
        for _ in range(n_repeats):
            model = PolynomialRegression(
                degree=degree, method=method, penalty=0.01,
                lr=0.01, n_iter=2000,
            )
            scores = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
            rows.append({"method": method, **scores})
    return pd.DataFrame(rows)


# ---------- Bonus: fitted curves at varying degrees ----------
def fit_curves(degrees=(1, 3, 6, 12), method=DEFAULT_METHOD,
               n_samples=150, noise=0.4):
    X_tr, X_te, y_tr, y_te = _make_data_1d(n_samples=n_samples, noise=noise)
    x_grid = np.linspace(-2.5, 2.5, 400).reshape(-1, 1)
    y_clean = np.sin(1.5 * x_grid.ravel()) + 0.5 * x_grid.ravel()

    fits = {}
    for d in degrees:
        model = PolynomialRegression(degree=d, method=method, penalty=0.0)
        model.fit(X_tr, y_tr)
        fits[d] = model.predict(x_grid)
    return X_tr, y_tr, x_grid, y_clean, fits


# ---------- Plotting ----------
def plot_results(df_deg, df_n, df_lam, df_meth,
                 X_tr, y_tr, x_grid, y_clean, fits):
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Polynomial Regression — Performance & Bias-Variance Analysis",
                 fontsize=15, fontweight="bold", y=0.99)

    # 1. Degree sweep — train vs test R²
    ax = axes[0, 0]
    ax.plot(df_deg["degree"], df_deg["train_r2"], marker="o", label="train R²")
    ax.plot(df_deg["degree"], df_deg["test_r2"],  marker="o", label="test R²")
    ax.set(xlabel="polynomial degree", ylabel="R²",
           title="Bias-Variance vs Polynomial Degree")
    ax.legend(fontsize=9)

    # 2. Learning curve
    ax = axes[0, 1]
    ax.plot(df_n["n_samples"], df_n["train_r2"], marker="o", label="train R²")
    ax.plot(df_n["n_samples"], df_n["test_r2"],  marker="o", label="test R²")
    ax.set_xscale("log")
    ax.set(xlabel="n_samples", ylabel="R²",
           title="Learning Curve (degree = 8)")
    ax.legend(fontsize=9)

    # 3. Penalty sweep at high degree
    ax = axes[0, 2]
    ax.plot(df_lam["lambda"], df_lam["train_r2"], marker="o", label="train R²")
    ax.plot(df_lam["lambda"], df_lam["test_r2"],  marker="o", label="test R²")
    ax.set_xscale("log")
    ax.set(xlabel=r"$\lambda$", ylabel="R²",
           title=r"Regularization at High Degree (= 12)")
    ax.legend(fontsize=9)

    # 4. Solver method comparison — fit time distribution
    ax = axes[1, 0]
    sns.boxplot(data=df_meth, x="method", y="fit_time", ax=ax)
    ax.set(ylabel="fit time (s)", xlabel="",
           title="Solver Fit-Time Distribution (degree = 6)")
    ax.tick_params(axis="x", rotation=20)

    # 5. Fitted curves at varying degrees
    ax = axes[1, 1]
    ax.scatter(X_tr.ravel(), y_tr, s=18, alpha=0.5, color="black", label="train")
    ax.plot(x_grid.ravel(), y_clean, "k--", alpha=0.6, linewidth=1.4, label="ground truth")
    for d, y_pred in fits.items():
        ax.plot(x_grid.ravel(), y_pred, linewidth=1.6, label=f"degree {d}")
    ax.set(xlabel="x", ylabel="y", title="Fitted Curves at Varying Degrees")
    ax.set_ylim(y_tr.min() - 1, y_tr.max() + 1)
    ax.legend(fontsize=8, ncol=2)

    # 6. Test MSE vs degree
    ax = axes[1, 2]
    ax.plot(df_deg["degree"], df_deg["mse"], marker="o", color="crimson")
    ax.set_yscale("log")
    ax.set(xlabel="polynomial degree", ylabel="test MSE (log)",
           title="Test Error vs Degree")

    fig.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()


# ---------- Driver ----------
def evaluate_PolyReg():
    print("\n>>> Sweeping polynomial degree...")
    df_deg = sweep_degree()
    print("\n>>> Sweeping training-set size...")
    df_n = sweep_n_samples()
    print("\n>>> Sweeping ridge penalty at high degree...")
    df_lam = sweep_penalty()
    print("\n>>> Comparing solver methods...")
    df_meth = compare_methods()
    print("\n>>> Generating fitted curves...")
    X_tr, y_tr, x_grid, y_clean, fits = fit_curves()

    print("\n=== degree sweep ===\n", df_deg.round(4))
    print("\n=== n_samples sweep ===\n", df_n.round(4))
    print("\n=== lambda sweep ===\n", df_lam.round(4))
    print("\n=== method comparison (means) ===")
    print(df_meth.groupby("method").mean(numeric_only=True).round(6))

    plot_results(df_deg, df_n, df_lam, df_meth,
                 X_tr, y_tr, x_grid, y_clean, fits)
    return df_deg, df_n, df_lam, df_meth


if __name__ == "__main__":
    evaluate_PolyReg()
