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

    def predict(self):
        pass

    def __str__(self):
        return "Logistic Regression Classifier"
    
    def __repr__(self) -> str:
        return f"LogisticRegression()"
    
