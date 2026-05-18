"""
K-nearest neighbors (KNN) is a simple, instance-based learning algorithm used for classification and regression tasks. The KNN algorithm works by finding the K nearest neighbors to a given data point and making predictions based on the majority class (for classification) or the average value (for regression) of those neighbors.
"""
import numpy as np


class KNN:
    def __init__(self, k:int=3, distance:str="euclidean"):
        # assertions
        assert k > 0 and isinstance(k, int), "k must be a positive integer"
        assert distance in ["euclidean", "manhattan", "cosine"], "Invalid distance metric"

        self.k = k
        self.distance = distance

    def _distance(self, x1:np.ndarray, x2:np.ndarray)->float:
            if self.distance == "euclidean":
                return np.sqrt(np.sum((x1 - x2) ** 2))
            elif self.distance == "manhattan":
                return np.sum(np.abs(x1 - x2))
            elif self.distance == "cosine":
                return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
            else:
                raise ValueError("Invalid distance metric")

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X:np.ndarray)->np.ndarray:
        X = np.atleast_2d(X)
        predictions = []
        for x in X:
            dists = np.array([self._distance(x,x_tr) for x_tr in self.X_train]) 

            k_indices = np.argsort(dists)[:self.k]

            predictions.append(np.bincount(self.y_train[k_indices]).argmax())

        return np.array(predictions)

    def __str__(self):
        return f"KNN(k={self.k})"
    
    def __repr__(self):
        return self.__str__()
