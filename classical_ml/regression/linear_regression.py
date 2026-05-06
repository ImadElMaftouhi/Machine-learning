import numpy as np

np.random.seed(42)

class LinearRegression:
    def __init__(self, solver="normal", lr=0.001, n_iter=1000):
        assert solver in ["normal", "gradient_descent", "auto"], "solver should be in (normal, gradient_descent, auto)"
        self.solver = solver
        self.lr = lr
        self.n_iter = n_iter
        self.coef_ = None
        self.loss_history = []
        self.intercept_ = None

    def __str__(self):
        return f"LinearRegression Algorithm"

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X]) # add w[0] as intercept
    
    def fit(self,X:np.ndarray,y:np.ndarray):
        X_b = self._add_bias(X)
        if self.solver == "normal":
            self.coef_ = np.linalg.pinv(X_b.T @ X_b) @  X_b.T @ y
        elif self.solver =="gradient_descent":
            n = X_b.shape[0]
            self.coef_ = np.zeros(X_b.shape[1])
            for _ in range(self.n_iter):
                residuals = X_b @ self.coef_ - y
                grad = (2/n) * X_b.T @ residuals       
                self.coef_ -= self.lr * grad
                self.loss_history.append(float(residuals @ residuals))
        if self.coef_ is not None: self.intercept_ = self.coef_[0]
        return self
        
    def predict(self, X):
        X_b = self._add_bias(X)
        if self.coef_ is None:
            raise ValueError("No weights computed. Try fitting the algorithm using LinearRegression.fit()")
        return X_b @ self.coef_