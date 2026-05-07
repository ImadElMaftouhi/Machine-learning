import numpy as np
import scipy.linalg as la


class RidgeRegression:
    def __init__(self, solver="auto", method:str="cholesky", penalty:float=0.01,lr=0.01, n_iter=2000, tol=1e-6):
        assert solver in ["normal", "gradient_descent", "auto"], f"solver argument must be in ('normal', 'gradient','auto')"
        self.solver = solver
        self.method = method
        self.penalty = penalty
        self.lr = lr
        self.n_iters = n_iter
        self.tol = tol
        self.coef_ = None
        self.intercept = None
        self.identity = None
        self.loss_history = []
    
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    def _add_bias(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def fit(self, X: np.ndarray, y: np.ndarray, ) -> "Ridge":
        self.loss_history = []
        X_aug = self._add_bias(X)
        y = np.asarray(y).flatten()
        n, d = X_aug.shape

        # Don't regularize the bias column (assume it's column 0).
        I = np.eye(d)
        I[0, 0] = 0.0

        solver = self.solver
        if solver == "auto":
            solver = "normal" if d <= 1200 else "gradient_descent"

        if solver == "normal":
            A = X_aug.T @ X_aug + self.penalty * I
            b = X_aug.T @ y
            if self.method == "cholesky":
                A = X_aug.T @ X_aug + self.penalty * I
                b = X_aug.T @ y
                self.coef_ = la.solve(A, b, assume_a="pos")

            elif self.method == "qr":
                # Stack to turn ridge into an OLS problem, then QR.
                sqrt_lam = np.sqrt(self.penalty)
                X_stack = np.vstack([X_aug, sqrt_lam * I])          # ((n+d), d)
                y_stack = np.concatenate([y, np.zeros(d)])
                Q, R = np.linalg.qr(X_stack, mode="reduced")
                self.coef_ = la.solve_triangular(R, Q.T @ y_stack)

            elif self.method == "svd":
                # SVD of X (not X^T X). Gives the classic ridge filter.
                U, s, Vt = np.linalg.svd(X_aug, full_matrices=False)
                d_filter = s / (s**2 + self.penalty)                # shrinks small singular values
                self.coef_ = Vt.T @ (d_filter * (U.T @ y))

            else:
                raise ValueError(f"Unknown method: {self.method}")

            return self

        if solver == "gradient_descent":
            # Safe lr: theoretical Lipschitz bound  L = 2*(||X||²/n + λ)
            L = 2.0 * (np.linalg.norm(X_aug, "fro") ** 2 / n + self.penalty)
            lr = min(self.lr, 1.0 / L)

            self.coef_ = np.zeros(d)
            for _ in range(self.n_iters):
                residual = X_aug @ self.coef_ - y
                grad = (2.0 / n) * (X_aug.T @ residual) + 2.0 * self.penalty * (I @ self.coef_)
                self.coef_ -= lr * grad
                loss = (residual @ residual) / n + self.penalty * (self.coef_ @ (I @ self.coef_))
                if not np.isfinite(loss):
                    break
                self.loss_history.append(loss)
            return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        return self._add_bias(X) @ self.coef_

    def evaluate(self,X,y):
        pass
