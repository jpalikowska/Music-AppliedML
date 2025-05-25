import numpy as np

class Optimizer:
    """
    Base optimizer class.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """
        Update parameters based on gradients.
        This method should be overridden by subclasses.

        Parameters:
        - params: list or dict of parameters (e.g., weights)
        - grads: list or dict of gradients (same structure as params)
        """
        raise NotImplementedError("`update` must be implemented by the subclass.")


class GDOptimizer(Optimizer):
    """
    Gradient descent optimizer.
    """
    def update(self, params, grads):
        for key in params:
            #params[key][:] -= self.learning_rate*grads[key]
            np.subtract(params[key], self.learning_rate * grads[key], out=params[key])