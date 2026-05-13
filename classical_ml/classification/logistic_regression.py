# classification/logistic_regression.py
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.weights = None
        self.bias = None
        self.learning_rate = lr
        self.n_iter = n_iter

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses_ = []

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
            self.losses_.append(loss)

            dw = (1 / n_samples) * X.T @ (y_hat - y)
            db = (1 / n_samples) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __str__(self):
        return "Logistic Regression Classifier"
    
    def __repr__(self) -> str:
        return f"LogisticRegression(lr={self.learning_rate}, n_iter={self.n_iter})"
    
