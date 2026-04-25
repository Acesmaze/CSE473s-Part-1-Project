"""
layers.py
---------
Core layer abstractions for the neural network library.

Classes
-------
Layer   -- Abstract base class; all layers inherit from this.
Dense   -- Fully-connected (linear) layer with weights W and bias b.
"""

import numpy as np


class Layer:
    """
    Abstract base class for all layers.

    Every concrete layer must override `forward` and `backward`.
    Layers that have trainable parameters must also override
    `get_params_and_grads` so the optimizer can update them.
    """

    def forward(self, x):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : np.ndarray  -- input activations

        Returns
        -------
        np.ndarray -- output activations
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Compute the backward pass (backpropagation).

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of the loss with respect to this layer's output (dL/dZ).

        Returns
        -------
        np.ndarray -- gradient with respect to this layer's input (dL/dX)
        """
        raise NotImplementedError

    def get_params_and_grads(self):
        """
        Return a list of (parameter_array, gradient_array) tuples.
        Used by the optimizer to update weights.
        Layers without parameters return an empty list.
        """
        return []


class Dense(Layer):
    """
    Fully-connected (Dense) layer.

    Forward:   Z = X @ W + b
    Backward:
        dL/dW = X^T  @ dL/dZ
        dL/db = sum(dL/dZ, axis=0)
        dL/dX = dL/dZ @ W^T

    Note: No extra 1/N division in backward — the loss function already
    normalises by the number of elements, so adding another 1/N here
    would double-count the averaging and break gradient checking.

    Parameters
    ----------
    in_features  : int   -- number of input neurons
    out_features : int   -- number of output neurons
    seed         : int or None -- optional RNG seed for reproducibility
    """

    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        # He initialisation: good default for ReLU, fine for Tanh/Sigmoid
        scale = np.sqrt(2.0 / in_features)
        self.W  = rng.standard_normal((in_features, out_features)) * scale
        self.b  = np.zeros((1, out_features))

        # Gradients — populated during backward()
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache: input stored during forward for use in backward
        self._x = None

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Parameters
        ----------
        x : ndarray of shape (N, in_features)

        Returns
        -------
        ndarray of shape (N, out_features)
        """
        self._x = x
        return x @ self.W + self.b

    # ------------------------------------------------------------------
    def backward(self, grad_output):
        """
        Parameters
        ----------
        grad_output : ndarray of shape (N, out_features)
            dL/dZ passed back from the next layer.

        Returns
        -------
        ndarray of shape (N, in_features) -- dL/dX
        """
        self.dW = self._x.T @ grad_output          # (in, out)
        self.db = grad_output.sum(axis=0, keepdims=True)  # (1, out)
        return grad_output @ self.W.T              # (N, in)

    # ------------------------------------------------------------------
    def get_params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]
