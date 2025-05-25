import numpy as np
from .base import TrainableModel


class LinearRegression(TrainableModel):
    """Linear regression model."""

    def __init__(self, w, b, optimizer):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = X.T @ residual / len(X)
        grad_b = np.mean(residual)
        return {"w": grad_w, "b": grad_b}
    
    def decision_function(self, X):
        return X @ self.w + self.b
    
    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return self.decision_function(X)
    

class LinearBinaryClassification(TrainableModel):
    """Linear binary classification model."""

    def __init__(self, w, b, optimizer,class_labels=[-1,1]):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.class_labels = [min(class_labels),max(class_labels)]
        self.threshold = (self.class_labels[1] - self.class_labels[0])/2.

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = X.T @ residual / len(X)
        grad_b = np.mean(residual)
        return {"w": grad_w, "b": grad_b}
    
    def decision_function(self, X):
        return X @ self.w + self.b
    
    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.where(self.decision_function(X)>=self.threshold, self.class_labels[1], self.class_labels[0])
    

class RidgeClassifier(LinearBinaryClassification):
    """
    Ridge binary classifier
    """
    def __init__(self, w, b, optimizer, lam=0.1, class_labels=[-1,1]):
        super().__init__(w, b, optimizer, class_labels)
        self.lam = lam

    def loss_grad(self, X,Y):
        """Loss gradient"""
        residual = self.decision_function(X) - Y
        w_grad = X.T@residual/X.shape[0] + 2*self.lam*self.w 
        b_grad = np.mean(residual, axis=0)
        return w_grad, b_grad
    

class LassoClassifier(LinearBinaryClassification):
    """
    Lasso binary classifier
    """
    def __init__(self, w, b, optimizer, lam=0.1, class_labels=[-1,1]):
        super().__init__(w, b, optimizer, class_labels)
        self.lam = lam

    def loss_grad(self, X,Y):
        """Loss gradient"""
        residual = self.decision_function(X) - Y
        w_grad = X.T@residual/X.shape[0] + self.lam*np.sign(self.w)
        b_grad = np.mean(residual, axis=0)
        return w_grad, b_grad
    

    