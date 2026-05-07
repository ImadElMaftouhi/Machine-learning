# linreg_analysis.py
# This code evaluates the performance of a custom Linear Regression implementation across varying numbers of features.
# the goal is to study the performance of the normal equation solver as the number of features increases, in terms of fit time, predict time, and various regression metrics (MSE, MAE, RMSE, R²).

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from timeit import repeat
from time import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np


np.random.seed(42)

def evaluate_linreg(solver="normal"):

    fit_time_history=[]
    r2_history = []
    mse_history = []
    mae_history = []
    rmse_history = []

    # print(f"{'='*30}")
    if solver=="normal":
        features_range = range(1, 101, 1)
        n_iter = 1000
    elif solver=="gradient_descent":
        features_range = range(1, 101, 1)
        n_iter = 1000
    else:
        features_range = range(1, 101, 1)
        n_iter=1000

    regressor = LinearRegression(solver=solver, lr=0.001, n_iter=n_iter)
    for i in features_range:
        if i//10: print(f"- Iterations N°{i}")

        X, y, true_coef = make_regression(
            n_samples=1000, n_features=i, n_informative=max(1, i - int(np.ceil(i / 2))),
            noise=15.0, coef=True, random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        t = repeat(lambda: regressor.fit(X_train, y_train), number=1, repeat=5)
        fit_time = np.mean(t)
        fit_time_history.append(fit_time)

        y_pred = regressor.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        r2_history.append(r2)
        mse_history.append(mse)
        mae_history.append(mae)
        rmse_history.append(rmse)
        if solver == "gradient_descent":
            if (abs(regressor.loss_history[-1] - regressor.loss_history[-10]) < 1e-6 or 
        regressor.loss_history[-1] > regressor.loss_history[-10]):
                print(f"- ALERT: GD did not converge for {i} features!")

    df = pd.DataFrame({
        'n_features': features_range,
        'fit_time': fit_time_history,
        'mse': mse_history,
        'mae': mae_history,
        'rmse': rmse_history,
        'r2': r2_history
    })

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sns.lineplot(data=df, x='n_features', y='fit_time', ax=axes[0,0], marker='o').set(title="Fit time O(p³/3)", ylabel="seconds")
    sns.lineplot(data=df, x='n_features', y='mse', ax=axes[0,1], marker='o').set(title="MSE", ylabel="MSE")
    sns.lineplot(data=df, x='n_features', y='mae', ax=axes[0,2], marker='o').set(title="MAE", ylabel="MAE")
    sns.lineplot(data=df, x='n_features', y='rmse', ax=axes[1,0], marker='o').set(title="RMSE", ylabel="RMSE")
    sns.lineplot(data=df, x='n_features', y='r2', ax=axes[1,1], marker='o').set(title="R²", ylabel="R²")
    plt.tight_layout()
    plt.show()

    return df


def compare_solvers():
    records = []
    features_range = [1, 10, 50, 100, 1000, 5000, 10000]

    for solver in ["normal", "gradient_descent"]:
        regressor = LinearRegression(solver=solver, lr=0.05, n_iter=1400)
        
        for n_feat in features_range:
            print(f"- Solver '{solver}' \t - n_feat = {n_feat}")
            X, y, _ = make_regression(n_samples=5000, n_features=n_feat, 
                                      n_informative=max(1, n_feat-1), 
                                      noise=15.0, coef=True, random_state=42)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)          # ← Fixed

            # Time the fit
            times = repeat(lambda: regressor.fit(X_train, y_train), number=1, repeat=5)
            fit_time = np.mean(times)

            y_pred = regressor.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            if solver == "gradient_descent":
                if (abs(regressor.loss_history[-1] - regressor.loss_history[-10]) < 1e-6 or 
                    regressor.loss_history[-1] > regressor.loss_history[-10]):
                    print(f"- ALERT: GD did not converge for {n_feat} features!")

            records.append([solver, n_feat, fit_time, mse, mae, rmse, r2])

    return pd.DataFrame(records, columns=["solver", "n_features", "fit_time", "mse", "mae", "rmse", "r2"])

           
        


if __name__=="__main__":
    
    # perf_linreg_normal = evaluate_linreg(solver="normal")
    # perf_linreg_GD = evaluate_linreg(solver="gradient_descent")

    # perf_linreg_normal["solver"] = "normal"
    # perf_linreg_GD["solver"] = "gradient_descent"
    # combined = pd.concat([perf_linreg_normal, perf_linreg_GD])

    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # sns.lineplot(data=combined, x="n_features", y="fit_time", hue="solver", ax=axes[0])
    # sns.lineplot(data=combined, x="n_features", y="r2", hue="solver", ax=axes[1])
    # plt.tight_layout()
    # plt.show()

    df = compare_solvers()
    print("\n", df)

    metrics = ['fit_time', 'mse', 'mae', 'rmse', 'r2']
    titles = ['Fit Time (s)', 'MSE', 'MAE', 'RMSE', 'R²']
    palette = {'normal': '#2196F3', 'gradient_descent': '#FF5722'}
    feature_vals = sorted(df['n_features'].unique())

    sns.set_style("whitegrid")
    sns.set_context("notebook")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Solver Comparison: Normal Equations vs Gradient Descent",
                 fontsize=15, fontweight='bold', y=1.01)

    for ax, metric, title in zip(axes.flat, metrics, titles):
        sns.lineplot(data=df, x='n_features', y=metric, hue='solver',
                     marker='o', markersize=7, linewidth=2,
                     palette=palette, ax=ax)

        # Annotate each point with its value
        for solver, grp in df.groupby('solver'):
            color = palette[solver]
            for _, row in grp.iterrows():
                val = float(row[metric])
                label = f"{val:.2g}" if abs(val) < 1000 else f"{val:.0f}"
                ax.annotate(label, xy=(row['n_features'], val),
                            xytext=(0, 7), textcoords='offset points',
                            ha='center', fontsize=7, color=color)

        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(title)
        ax.set_xscale('log')
        ax.set_xticks(feature_vals)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Solver', fontsize=9)

    # 6th panel: GD / Normal fit-time ratio — reveals where each solver wins
    ax6 = axes[1, 2]
    normal_t = df[df['solver'] == 'normal'].set_index('n_features')['fit_time']
    gd_t = df[df['solver'] == 'gradient_descent'].set_index('n_features')['fit_time']
    ratio = (gd_t / normal_t).reset_index()
    ratio.columns = ['n_features', 'ratio']

    colors = ['#4CAF50' if r < 1 else '#F44336' for r in ratio['ratio']]
    ax6.bar(range(len(ratio)), ratio['ratio'], color=colors, alpha=0.8, edgecolor='white')
    ax6.axhline(1, color='gray', linestyle='--', linewidth=1.2, label='Parity (ratio = 1)')
    for i, row in ratio.iterrows():
        ax6.text(i, row['ratio'] + 0.03, f"{row['ratio']:.2f}x",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax6.set_xticks(range(len(ratio)))
    ax6.set_xticklabels(ratio['n_features'], rotation=45)
    ax6.set_title("Fit Time Ratio (GD / Normal)", fontweight='bold', fontsize=12)
    ax6.set_xlabel("Number of Features")
    ax6.set_ylabel("Ratio  (< 1 → GD faster,  > 1 → Normal faster)")
    ax6.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    