# performance_evaluation.py
# This code evaluates the performance of a custom Linear Regression implementation across varying numbers of features.
# the goal is to study the performance of the normal equation solver as the number of features increases, in terms of fit time, predict time, and various regression metrics (MSE, MAE, RMSE, R²).

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression
from sklearn.datasets import make_regression
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from timeit import repeat
from time import time

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd
import numpy as np


np.random.seed(42)

def evaluate_linreg_normal():
    fit_time_history=[]
    predict_time_history=[]
    r2_history = []
    mse_history = []
    mae_history = []
    rmse_history = []

    for i in range(1, 101):
        # print(f"{'='*30}")
        regressor = LinearRegression(solver="normal", lr=0.001, n_iter=1000)
        print(f"- Iterations N°{i}")
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

        start = time()
        y_pred = regressor.predict(X_test)
        predict_time = time() - start
        predict_time_history.append(predict_time)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        r2_history.append(r2)
        mse_history.append(mse)
        mae_history.append(mae)
        rmse_history.append(rmse)


    # fig, ax = plt.subplots(1, 6, figsize=(25, 4))
    # ax[0].plot(range(1, 101), fit_time_history)
    # ax[0].set(title="Fit time vs n_features O(p^3 + np^2)", xlabel="n_features", ylabel="seconds")

    # ax[1].plot(range(1, 101), predict_time_history)
    # ax[1].set(title="Predict time vs n_features O(n*p)", xlabel="n_features", ylabel="seconds")

    # ax[2].plot(range(1, 101), mse_history)
    # ax[2].set(title="MSE History", xlabel="n_features", ylabel="MSE")

    # ax[3].plot(range(1, 101), mae_history)
    # ax[3].set(title="MAE vs n_features", xlabel="n_features", ylabel="MAE")

    # ax[4].plot(range(1, 101), rmse_history)
    # ax[4].set(title="RMSE vs n_features", xlabel="n_features", ylabel="RMSE")

    # ax[5].plot(range(1, 101), r2_history)
    # ax[5].set(title="R² vs n_features", xlabel="n_features", ylabel="R²")
    # plt.tight_layout()
    # plt.show()

    df = pd.DataFrame({
        'n_features': range(1, 101),
        'fit_time': fit_time_history,
        'predict_time': predict_time_history,
        'mse': mse_history,
        'mae': mae_history,
        'rmse': rmse_history,
        'r2': r2_history
    })

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sns.lineplot(data=df, x='n_features', y='fit_time', ax=axes[0,0], marker='o').set(title="Fit time O(p³+np²)", ylabel="seconds")
    sns.lineplot(data=df, x='n_features', y='predict_time', ax=axes[0,1], marker='o').set(title="Predict time O(n*p)", ylabel="seconds")
    sns.lineplot(data=df, x='n_features', y='mse', ax=axes[0,2], marker='o').set(title="MSE", ylabel="MSE")
    sns.lineplot(data=df, x='n_features', y='mae', ax=axes[1,0], marker='o').set(title="MAE", ylabel="MAE")
    sns.lineplot(data=df, x='n_features', y='rmse', ax=axes[1,1], marker='o').set(title="RMSE", ylabel="RMSE")
    sns.lineplot(data=df, x='n_features', y='r2', ax=axes[1,2], marker='o').set(title="R²", ylabel="R²")
    plt.tight_layout()
    plt.show()

    # fig = make_subplots(rows=2, cols=3, subplot_titles=[
    #     "Fit time O(p³ + np²)", "Predict time O(n*p)", 
    #     "MSE", "MAE", "RMSE", "R²"
    # ])

    # metrics = [fit_time_history, predict_time_history, mse_history, 
    #            mae_history, rmse_history, r2_history]
    # titles = ["seconds", "seconds", "MSE", "MAE", "RMSE", "R²"]

    # for i, (data, title) in enumerate(zip(metrics, titles)):
    #     row, col = divmod(i, 3)
    #     fig.add_trace(go.Scatter(x=list(range(1,101)), y=data, mode='lines+markers'), 
    #                   row=row+1, col=col+1)
    #     fig.update_yaxes(title_text=title, row=row+1, col=col+1)

    # fig.update_layout(height=600, width=1200, title_text="Performance vs n_features",
    #                   template="plotly_white", showlegend=False)
    # fig.show()

    # alt.data_transformers.disable_max_rows()
    # charts = []
    # metrics = ['fit_time', 'predict_time', 'mse', 'mae', 'rmse', 'r2']
    # titles = ["Fit time O(p³+np²)", "Predict time O(n*p)", "MSE", "MAE", "RMSE", "R²"]
    # ylabels = ["seconds", "seconds", "MSE", "MAE", "RMSE", "R²"]
    # for metric, title, ylabel in zip(metrics, titles, ylabels):
    #     chart = alt.Chart(df).mark_line(point=True).encode(
    #         x='n_features',
    #         y=alt.Y(metric, title=ylabel),
    #         tooltip=['n_features', metric]
    #     ).properties(title=title, width=320, height=220)
    #     charts.append(chart)

    # # This forces display
    # final_chart = alt.concat(*charts, columns=3)
    # final_chart.display()

    # plots = []
    # metrics = ['fit_time', 'predict_time', 'mse', 'mae', 'rmse', 'r2']
    # titles = ["Fit time O(p³+np²)", "Predict time O(n*p)", "MSE", "MAE", "RMSE", "R²"]
    # ylabels = ["seconds", "seconds", "MSE", "MAE", "RMSE", "R²"]
    # for metric, title, ylabel in zip(metrics, titles, ylabels):
    #     p = figure(title=title, width=350, height=250, x_axis_label='n_features', y_axis_label=ylabel)
    #     p.line(df['n_features'], df[metric], line_width=2)
    #     p.circle(df['n_features'], df[metric], size=4)
    #     plots.append(p)
    # grid = gridplot([plots[:3], plots[3:]], sizing_mode='scale_width')
    # show(grid)


if __name__=="__main__":
    
    # accepting argument instead of writing a whole menu
    evaluate_linreg_normal()
