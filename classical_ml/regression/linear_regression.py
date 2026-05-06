import numpy as np
import scipy.linalg as la

class LinearRegression:
    def __init__(self, solver="auto", lr=0.001, n_iter=1000, tol=1e-6):
        # Use ValueError instead of assert for user-facing validation
        if solver not in ["normal", "gradient_descent", "auto"]:
            raise ValueError("solver should be in (normal, gradient_descent, auto)")
            
        self.solver = solver
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.coef_ = None
        self.loss_history = []
        self.intercept_ = None

    def __str__(self):
        return f"LinearRegression(solver={self.solver})"

    def _add_bias(self, X):
        # Ensure X is 2D even if a 1D array is passed
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def fit(self, X, y):
        if X is None or y is None: raise ValueError("X and y cannot be None")
        self.loss_history = []
        X_b = self._add_bias(X)
        
        y = np.asarray(y).flatten() # Ensure y is 1D
        
        # Auto-select solver based on data size
        if self.solver == "auto":
            # Standard heuristic: Normal equation is faster for small/medium features
            self.solver = "normal" if X_b.shape[1] < 10000 else "gradient_descent"

        if self.solver == "normal":
            # VASTLY superior numerical stability compared to X.T @ X
            # lstsq uses SVD and handles rank-deficient matrices natively
            self.coef_, _, _, _ = la.lstsq(X_b, y)
            
        elif self.solver == "gradient_descent":
            n = X_b.shape[0]
            self.coef_ = np.zeros(X_b.shape[1])
            
            for i in range(self.n_iter):
                residuals = X_b @ self.coef_ - y
                grad = (2 / n) * (X_b.T @ residuals)
                self.coef_ -= self.lr * grad
                
                loss = float(residuals @ residuals)
                self.loss_history.append(loss)
                
                # Early stopping mechanism
                if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                    break

        if self.coef_ is not None:
            self.intercept_ = self.coef_[0]
            
        return self
        
    def predict(self, X):
        X_b = self._add_bias(X)
        if self.coef_ is None:
            raise ValueError("This LinearRegression instance is not fitted yet. Call 'fit' first.")
        return X_b @ self.coef_