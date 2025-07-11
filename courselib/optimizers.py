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
    Gradient descent optimizer with optional learning rate schedule.

    Parameters:
    - learning_rate (float): Initial learning rate
    - schedule_fn (callable): Function(step) → new_learning_rate
    """

    def __init__(self, learning_rate=0.01, schedule_fn=None):
        super().__init__(learning_rate)
        self.schedule_fn = schedule_fn
        self.step = 0

    def update(self, params, grads):
        if self.schedule_fn is not None:
            self.step += 1
            self.learning_rate = self.schedule_fn(self.step)

        for key in params:
            np.subtract(params[key], self.learning_rate * grads[key], out=params[key])

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(grads[key])
                self.v[key] = np.zeros_like(grads[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)