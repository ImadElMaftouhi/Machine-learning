import numpy as np



class RidgeRegression:
    def __init__(self, solver="auto", penalty:float=0.01,lr=0.01, n_iter=2000, tol=1e-6):
        self.solver = solver
        self.penalty = penalty
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.coef_ = None
        self.intercept = None
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
    
    def fit(self, X:np.ndarray, y=np.ndarray)->None:
        pass

    def predict(self,X,y):
        pass

    def evaluate(self,X,y):
        pass
