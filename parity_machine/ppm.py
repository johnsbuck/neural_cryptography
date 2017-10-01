import numpy as np
from utils import learning_rules as rules


class PermutationParityMachine(object):
    """

    """

    def __init__(self, k, n):
        """

        """
        # Hyper-parameters
        self.k = k
        self.n = n

        # Training Parameters
        self._sigma = None
        self._x = None
        self._tau = None

        self.weights = np.random.randint(0, 2, [k, n])

    def output(self, x):
        """

        Args:
            x:

        Returns:

        """
        self._sigma = np.sign(np.sum(np.bitwise_xor(x, self.weights), axis=0) - self.n/2)
        self._sigma[np.where(self._sigma < 0)] = 0
        self._tau = np.bitwise_xor.reduce(self._sigma)

        self._sigma = self._sigma.reshape(-1, 1)

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

        self.weights = rules.hebbian(self.weights, self._x, self._sigma, 0, 1, self._tau, tau_b)

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
