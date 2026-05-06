# performance_evaluation.py
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


if __name__=="__main__":
    
    # accepting argument instead of writing a whole menu
    perf_linreg_normal = evaluate_linreg(solver="normal")
    perf_linreg_GD = evaluate_linreg(solver="gradient_descent")

    perf_linreg_normal["solver"] = "normal"
    perf_linreg_GD["solver"] = "gradient_descent"
    combined = pd.concat([perf_linreg_normal, perf_linreg_GD])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=combined, x="n_features", y="fit_time", hue="solver", ax=axes[0])
    sns.lineplot(data=combined, x="n_features", y="r2", hue="solver", ax=axes[1])
    plt.tight_layout()
    plt.show()

    