import numpy as np
from .base import TrainableModel

class LinearSVM(TrainableModel):

    def __init__(self, w, b, optimizer, C=10.):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.C = C
    
    def loss_grad(self, X, y):
       # Compute raw model output
        output = self.decision_function(X)

        # Identify margin violations: where 1 - y*h(x) > 0
        mask = (1 - y * output) > 0
        y_masked = y[mask]
        X_masked = X[mask]

        # Compute 
        if len(y_masked) > 0:
            grad_w = 2 * self.w - self.C * np.mean(y_masked[:, None] * X_masked, axis=0)
            grad_b = - self.C * np.mean(y_masked)
        else:
            grad_b = 0.0
            grad_w = 2 * self.w

        return {"w": grad_w, "b": grad_b}
    
    def decision_function(self, X):
        return X @ self.w + self.b
    
    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
