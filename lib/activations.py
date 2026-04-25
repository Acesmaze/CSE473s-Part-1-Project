"""
activations.py
--------------
Activation functions implemented as Layer subclasses.

All activations have no trainable parameters; get_params_and_grads()
is inherited from Layer and returns [].

Classes
-------
ReLU     -- max(0, x)
Sigmoid  -- 1 / (1 + exp(-x))
Tanh     -- (exp(x) - exp(-x)) / (exp(x) + exp(-x))
Softmax  -- exp(x_i) / sum(exp(x_j))   [row-wise]
Dropout  -- Inverted dropout regularisation
"""

import numpy as np
from lib.layers import Layer


class ReLU(Layer):
    """Rectified Linear Unit:  f(x) = max(0, x)"""

    def forward(self, x):
        self._mask = (x > 0)
        return np.where(self._mask, x, 0.0)

    def backward(self, grad_output):
        # Gradient is 1 where x > 0, else 0
        return grad_output * self._mask


class Sigmoid(Layer):
    """
    Sigmoid:  f(x) = 1 / (1 + exp(-x))

    Uses a numerically stable implementation to avoid overflow for
    large negative x values.
    """

    def forward(self, x):
        # Stable: for x>=0 use 1/(1+exp(-x)); for x<0 use exp(x)/(1+exp(x))
        self._out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )
        return self._out

    def backward(self, grad_output):
        s = self._out
        return grad_output * s * (1.0 - s)


class Tanh(Layer):
    """
    Hyperbolic Tangent:  f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Derivative:  f'(x) = 1 - f(x)^2
    """

    def forward(self, x):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad_output):
        return grad_output * (1.0 - self._out ** 2)


class Softmax(Layer):
    """
    Row-wise Softmax:  f(x)_i = exp(x_i) / sum_j(exp(x_j))

    Numerically stable: subtract row-wise max before exponentiation.

    Note: When used with Cross-Entropy loss the combined gradient
    simplifies to (y_pred - y_true)/N. This backward implements the
    full Jacobian-vector product for standalone use.
    """

    def forward(self, x):
        # Subtract max per row for numerical stability
        shifted = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self._out = exp_x / exp_x.sum(axis=1, keepdims=True)
        return self._out

    def backward(self, grad_output):
        # dL/dx_i = s_i * (dL/ds_i - sum_j(s_j * dL/ds_j))
        s = self._out
        dot = (grad_output * s).sum(axis=1, keepdims=True)
        return s * (grad_output - dot)


class Dropout(Layer):
    """
    Inverted Dropout regularisation.

    During training, each neuron is independently zeroed with
    probability `rate`. Surviving activations are scaled up by
    1/(1-rate) so that the expected output is unchanged — this
    means inference requires no special scaling.

    Parameters
    ----------
    rate     : float  -- fraction of neurons to DROP (e.g. 0.3 → 30%)
    """

    def __init__(self, rate=0.5):
        self.rate     = rate
        self.training = True   # set to False at inference time
        self._mask    = None

    def forward(self, x):
        if not self.training:
            return x
        # Inverted dropout mask: 1/(1-rate) for surviving neurons, 0 otherwise
        self._mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
        return x * self._mask

    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self._mask
