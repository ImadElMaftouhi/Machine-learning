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

        for _ in range(self.n_iter):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            dw = (1 / n_samples) * X.T @ (y_hat - y)
            db = (1 / n_samples) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

    def predict(self):
        pass

    def __str__(self):
        return "Logistic Regression Classifier"
    
    def __repr__(self) -> str:
        return f"LogisticRegression()"
    
