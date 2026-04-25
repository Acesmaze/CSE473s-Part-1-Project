"""
network.py
----------
Sequential neural network model.

Classes
-------
Sequential -- Stacks layers, orchestrates forward/backward, and training.
"""

import numpy as np
from lib.activations import Dropout


class Sequential:
    """
    A linear stack of layers.

    Usage
    -----
    model = Sequential([Dense(2, 4), Tanh(), Dense(4, 1), Sigmoid()])
    history = model.train(X, y, loss_fn=mse, optimizer=opt, epochs=1000)
    preds = model.predict(X)

    Parameters
    ----------
    layers : list of Layer objects
    """

    def __init__(self, layers):
        self.layers = layers

    # ------------------------------------------------------------------
    def forward(self, x):
        """Run the full forward pass and return the network's output."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ------------------------------------------------------------------
    def backward(self, grad):
        """
        Run the full backward pass.

        Parameters
        ----------
        grad : ndarray -- dL/d(output), returned by the loss function.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # ------------------------------------------------------------------
    def get_all_params_and_grads(self):
        """Collect (param, grad) pairs from every layer for the optimizer."""
        pairs = []
        for layer in self.layers:
            pairs.extend(layer.get_params_and_grads())
        return pairs

    # ------------------------------------------------------------------
    def _set_training(self, training: bool):
        """Toggle training mode — currently affects Dropout layers."""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training

    # ------------------------------------------------------------------
    def train(self, X, y, loss_fn, optimizer, epochs=1000,
              batch_size=None, verbose=True, print_every=100):
        """
        Train the network.

        Parameters
        ----------
        X           : ndarray (N, features)  -- training inputs
        y           : ndarray (N, outputs)   -- training targets
        loss_fn     : callable(y_true, y_pred) -> (loss, grad)
        optimizer   : optimizer instance with .update(params_and_grads)
        epochs      : int   -- number of full passes over the data
        batch_size  : int or None  -- mini-batch size; None = full-batch GD
        verbose     : bool  -- print loss during training
        print_every : int   -- print interval (epochs)

        Returns
        -------
        history : list of float  -- loss recorded after each epoch
        """
        N = X.shape[0]
        history = []
        self._set_training(True)

        for epoch in range(1, epochs + 1):
            # ---- Build mini-batches ----
            if batch_size is None or batch_size >= N:
                batches = [(X, y)]
            else:
                idx = np.random.permutation(N)
                batches = [
                    (X[idx[i:i + batch_size]], y[idx[i:i + batch_size]])
                    for i in range(0, N, batch_size)
                ]

            epoch_loss = 0.0
            for Xb, yb in batches:
                # Forward
                y_pred = self.forward(Xb)
                # Loss + initial gradient
                loss, grad = loss_fn(yb, y_pred)
                epoch_loss += loss
                # Backward
                self.backward(grad)
                # Parameter update
                optimizer.update(self.get_all_params_and_grads())

            epoch_loss /= len(batches)
            history.append(epoch_loss)

            if verbose and epoch % print_every == 0:
                print(f"Epoch {epoch:5d} | Loss: {epoch_loss:.6f}")

        self._set_training(False)
        return history

    # ------------------------------------------------------------------
    def predict(self, X):
        """Run inference (Dropout disabled)."""
        self._set_training(False)
        return self.forward(X)

    # ------------------------------------------------------------------
    def encode(self, X, n_encoder_layers):
        """
        Partial forward pass through the first n_encoder_layers.
        Used to extract latent representations from an autoencoder.

        Parameters
        ----------
        X               : ndarray -- input data
        n_encoder_layers: int     -- how many layers constitute the encoder
        """
        self._set_training(False)
        out = X
        for layer in self.layers[:n_encoder_layers]:
            out = layer.forward(out)
        return out
