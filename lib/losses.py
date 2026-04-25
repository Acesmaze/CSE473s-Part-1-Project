"""
losses.py
---------
Loss functions for training neural networks.

Each function returns a tuple: (scalar_loss, dL/dy_pred)
so the gradient flows directly into model.backward().

Functions
---------
mse                  -- Mean Squared Error
binary_cross_entropy -- Binary Cross-Entropy (for sigmoid outputs)
"""

import numpy as np


def mse(y_true, y_pred):
    """
    Mean Squared Error.

    L = (1/N) * sum((y_pred - y_true)^2)

    Gradient w.r.t y_pred:
        dL/dy_pred = 2 * (y_pred - y_true) / N_total

    where N_total = number of elements = batch_size * output_dim.
    Note: dividing by N_total (not batch_size) keeps the gradient
    scale consistent regardless of output dimensionality.

    Parameters
    ----------
    y_true : ndarray of shape (N, D)
    y_pred : ndarray of shape (N, D)

    Returns
    -------
    loss : float
    grad : ndarray of shape (N, D)
    """
    diff = y_pred - y_true
    N    = y_true.size          # total elements (N * D)
    loss = np.mean(diff ** 2)
    grad = 2.0 * diff / N
    return loss, grad


def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy loss.

    L = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))

    Expects y_pred to already be sigmoid outputs in the range (0, 1).
    Clips predictions to [eps, 1-eps] to avoid log(0).

    Parameters
    ----------
    y_true : ndarray of shape (N, D)  -- binary labels {0, 1}
    y_pred : ndarray of shape (N, D)  -- sigmoid probabilities

    Returns
    -------
    loss : float
    grad : ndarray of shape (N, D)
    """
    eps = 1e-12
    p    = np.clip(y_pred, eps, 1.0 - eps)
    N    = y_true.size
    loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    grad = (-(y_true / p) + (1.0 - y_true) / (1.0 - p)) / N
    return loss, grad
