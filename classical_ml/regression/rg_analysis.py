# regression/rg_analysis.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from ridge_regression import RidgeRegression

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from time import perf_counter


VALID_SOLVERS = {"normal", "gradient_descent", "auto"}
VALID_METHODS = {"cholesky", "qr", "svd"}

# All (solver, method) pairs to benchmark. method=None means solver chooses internally.
ALL_CONFIGS = [
    ("normal", "cholesky"),
    ("normal", "qr"),
    ("normal", "svd"),
    ("gradient_descent", None),
    ("auto", "cholesky"),
]

def _config_label(solver, method):
    return solver if method is None else f"{solver}/{method}"


def _fit_and_score(model, X_train, y_train, X_test, y_test):
    t0 = perf_counter()
    model.fit(X_train, y_train)
    fit_time = perf_counter() - t0

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "fit_time": fit_time,
        "r2":   r2_score(y_test, y_pred),
        "mse":  mse,
        "rmse": np.sqrt(mse),
        "mae":  mean_absolute_error(y_test, y_pred),
    }


def _make_split(n_samples, n_features, noise=15.0, random_state=42):
    n_informative = max(1, n_features - int(np.ceil(n_features / 2)))
    X, y, true_coef = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, noise=noise,
        coef=True, random_state=random_state,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, true_coef


# ---------- Experiment 1: scale with number of features ----------
def sweep_n_features(solver="normal", method="cholesky",
                     feature_grid=(2, 5, 10, 25, 50, 100, 250, 500),
                     n_samples=1000):
    rows = []
    for d in feature_grid:
        X_tr, X_te, y_tr, y_te, _ = _make_split(n_samples, d)
        model = RidgeRegression(solver=solver, method=method,
                                penalty=0.01, lr=0.01, n_iter=1000)
        scores = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
        rows.append({"n_features": d, **scores})
    return pd.DataFrame(rows)


# ---------- Experiment 2: bias-variance via lambda sweep ----------
def sweep_penalty(solver="normal", method="cholesky",
                  lambdas=np.logspace(-4, 4, 25),
                  n_samples=300, n_features=100, noise=20.0):
    X_tr, X_te, y_tr, y_te, _ = _make_split(n_samples, n_features, noise=noise)
    rows = []
    for lam in lambdas:
        model = RidgeRegression(solver=solver, method=method,
                                penalty=lam, lr=0.01, n_iter=1000)
        scores = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
        # train-set score too, to see over/underfitting
        train_pred = model.predict(X_tr)
        rows.append({
            "lambda": lam,
            "train_r2": r2_score(y_tr, train_pred),
            **scores,
        })
    return pd.DataFrame(rows)


# ---------- Experiment 3: robustness to noise ----------
def sweep_noise(solver="normal", method="cholesky",
                noise_grid=(0.1, 1, 5, 10, 25, 50, 100),
                n_samples=1000, n_features=50):
    rows = []
    for noise in noise_grid:
        X_tr, X_te, y_tr, y_te, _ = _make_split(n_samples, n_features, noise=noise)
        model = RidgeRegression(solver=solver, method=method,
                                penalty=0.01, lr=0.01, n_iter=1000)
        scores = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
        rows.append({"noise": noise, **scores})
    return pd.DataFrame(rows)


# ---------- Experiment 4: compare all solvers/methods ----------
def compare_solvers(n_samples=1000, n_features=200, n_repeats=5):
    X_tr, X_te, y_tr, y_te, _ = _make_split(n_samples, n_features)
    configs = [
        ("normal", "cholesky"),
        ("normal", "qr"),
        ("normal", "svd"),
        ("gradient_descent", None),
    ]
    rows = []
    for solver, method in configs:
        for _ in range(n_repeats):
            model = RidgeRegression(solver=solver, method=method,
                                    penalty=0.01, lr=0.01, n_iter=1000)
            scores = _fit_and_score(model, X_tr, y_tr, X_te, y_te)
            label = solver if method is None else f"{solver}/{method}"
            rows.append({"config": label, **scores})
    return pd.DataFrame(rows)


# ---------- Plotting ----------
def plot_results(df_feat, df_lam, df_noise, df_solv):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Ridge Regression — Performance Across Solvers & Scenarios",
                 fontsize=14, fontweight="bold", y=0.99)

    # --- Fit time vs dimensionality ---
    ax = axes[0, 0]
    sns.lineplot(data=df_feat, x="n_features", y="fit_time",
                 hue="config", marker="o", ax=ax)
    ax.set(xlabel="n_features", ylabel="fit time (s)",
           title="Fit Time vs. Dimensionality")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(title="config", fontsize=8)

    # --- Bias-variance tradeoff: test vs train R² per config ---
    ax = axes[0, 1]
    for cfg, grp in df_lam.groupby("config"):
        line, = ax.plot(grp["lambda"], grp["r2"],   marker="o", label=f"{cfg} test")
        ax.plot(grp["lambda"], grp["train_r2"], marker="x",
                color=line.get_color(), linestyle="--", alpha=0.6, label=f"{cfg} train")
    ax.set(xlabel=r"$\lambda$", ylabel=r"$R^2$", title="Bias-Variance via Penalty")
    ax.set_xscale("log")
    ax.legend(fontsize=7, ncol=2)

    # --- Robustness to noise ---
    ax = axes[1, 0]
    sns.lineplot(data=df_noise, x="noise", y="rmse",
                 hue="config", marker="o", ax=ax)
    ax.set(xlabel="noise std", ylabel="RMSE", title="Robustness to Label Noise")
    ax.set_xscale("log")
    ax.legend(title="config", fontsize=8)

    # --- Solver / method fit-time distribution ---
    ax = axes[1, 1]
    sns.boxplot(data=df_solv, x="config", y="fit_time", ax=ax)
    ax.set(ylabel="fit time (s)", xlabel="", title="Solver / Method Fit-Time Distribution")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    plt.show()


def evaluate_RidgeReg():
    feat_frames, lam_frames, noise_frames = [], [], []

    for solver, method in ALL_CONFIGS:
        label = _config_label(solver, method)
        print(f"\n>>> Running config: {label}")

        df_f = sweep_n_features(solver, method)
        df_f["config"] = label
        feat_frames.append(df_f)

        df_l = sweep_penalty(solver, method)
        df_l["config"] = label
        lam_frames.append(df_l)

        df_n = sweep_noise(solver, method)
        df_n["config"] = label
        noise_frames.append(df_n)

    df_feat  = pd.concat(feat_frames,  ignore_index=True)
    df_lam   = pd.concat(lam_frames,   ignore_index=True)
    df_noise = pd.concat(noise_frames, ignore_index=True)
    df_solv  = compare_solvers()

    print("\n=== n_features sweep (means by config) ===")
    print(df_feat.groupby("config").mean(numeric_only=True).round(4))
    print("\n=== lambda sweep (means by config) ===")
    print(df_lam.groupby("config").mean(numeric_only=True).round(4))
    print("\n=== noise sweep (means by config) ===")
    print(df_noise.groupby("config").mean(numeric_only=True).round(4))
    print("\n=== solver comparison (means) ===")
    print(df_solv.groupby("config").mean(numeric_only=True).round(4))

    plot_results(df_feat, df_lam, df_noise, df_solv)
    return df_feat, df_lam, df_noise, df_solv


if __name__ == "__main__":
    evaluate_RidgeReg()
