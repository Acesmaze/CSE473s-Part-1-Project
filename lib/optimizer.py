"""
optimizer.py
------------
Optimizer implementations.

Classes
-------
SGDMomentum -- Stochastic Gradient Descent with Momentum
"""

import numpy as np


class SGDMomentum:
    """
    Stochastic Gradient Descent with Momentum.

    Update rule:
        v  = momentum * v  -  learning_rate * grad
        W += v

    The velocity term `v` accumulates a history of gradients,
    dampening oscillations and accelerating convergence in
    consistent gradient directions.

    Parameters
    ----------
    learning_rate : float  -- step size (default 0.01)
    momentum      : float  -- velocity decay factor, typically 0.9 (default 0.9)
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr       = learning_rate
        self.momentum = momentum
        # Dictionary mapping id(param) -> velocity array
        self._velocities = {}

    def update(self, params_and_grads):
        """
        Apply one optimisation step to all parameters.

        Parameters
        ----------
        params_and_grads : list of (param_ndarray, grad_ndarray) tuples
            Typically obtained from model.get_all_params_and_grads().
        """
        for param, grad in params_and_grads:
            key = id(param)
            if key not in self._velocities:
                self._velocities[key] = np.zeros_like(param)
            v = self._velocities[key]
            # In-place update preserves the reference held by the layer
            v[:]    = self.momentum * v - self.lr * grad
            param  += v
