import numpy as np
import utils.learning_rules as rules


class TreeParityMachine(object):
    """Used in Neural Cryptography to generate a key of size K*N via its weights.

    """

    def __init__(self, k, n, l):
        """

        Args:
            k: Integer. Number of hidden neurons in hidden layer.
            n: Integer. Number of weights for each neuron.
            l: Integer. Range of integers for weights from -l to l.
        """
        # Hyper-parameters
        self.k = k
        self.l = l
        self.n = n

        # Training Parameters
        self._sigma = None
        self._x = None
        self._tau = None

        self.weights = np.random.randint(-l, l+1, [k, n])

    def output(self, x):
        """

        Args:
            x:

        Returns:

        """
        self._sigma = np.sign(np.sum(x * self.weights, axis=1)).reshape(-1, 1)
        self._sigma[np.where(self._sigma == 0)] -= 1
        self._tau = np.prod(self._sigma)

        # Create copy of input
        self._x = np.empty_like(x)
        self._x[:] = x

        return self._tau

    def update(self, tau_b):
        """

        Args:
            tau_b:

        Returns:

        """
        if None is self._sigma or None is self._x or None is self._tau:
            raise ValueError("Did not obtain output for training. Training parameters undefined.")

        self.weights = rules.hebbian(self.weights, self._x, self._sigma, -self.l, self.l, self._tau, tau_b)

        # Reset training variables for next update
        self._sigma = None
        self._x = None
        self._tau = None

    def get_key(self):
        """ Returns the weights as a K*N integer array.

        Returns:
            Numpy Array (Int64).
        """
        return self.weights.reshape(self.k * self.n)