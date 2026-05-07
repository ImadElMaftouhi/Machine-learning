import numpy as np
import scipy.linalg as la
from numba import njit

class LinearRegression:
    def __init__(self, solver="auto", lr=0.01, n_iter=2000, tol=1e-6):
        if solver not in ["normal", "gradient_descent", "auto"]:
            raise ValueError("solver must be 'normal', 'gradient_descent', or 'auto'")
            
        self.solver = solver
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []

    def _add_bias(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    @njit(fastmath=True)
    def _gd_fit(X_b, y, lr, n_iter, tol):
        n_samples, n_features = X_b.shape
        coef = np.zeros(n_features)
        loss_history = np.zeros(n_iter)
        
        for i in range(n_iter):
            residuals = X_b @ coef - y
            grad = (2 / n_samples) * (X_b.T @ residuals)
            coef -= lr * grad
            
            loss = np.dot(residuals, residuals)
            loss_history[i] = loss
            
            if i > 10 and abs(loss_history[i] - loss_history[i-10]) < tol:
                return coef, loss_history[:i+1]
        
        return coef, loss_history

    def fit(self, X, y):
        self.loss_history = []
        X_b = self._add_bias(X)
        y = np.asarray(y).flatten()
        
        solver = self.solver
        if solver == "auto":
            solver = "normal" if X_b.shape[1] <= 1200 else "gradient_descent"

        if solver == "normal":
            self.coef_, _, _, _ = la.lstsq(X_b, y)
            
        elif solver == "gradient_descent":
            self.coef_, self.loss_history = self._gd_fit(
                X_b.astype(np.float64), 
                y.astype(np.float64), 
                self.lr, 
                self.n_iter, 
                self.tol
            )

        self.intercept_ = self.coef_[0]
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        return self._add_bias(X) @ self.coef_