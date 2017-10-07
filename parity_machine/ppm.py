import numpy as np
from utils import learning_rules as rules


class PermutationParityMachine(object):
    """Permutation Parity Machine (PPM)
    Used in Neural Cryptography to generate a key of size K*N via its weights. Weights consists of 0s and 1s.

    """

    def __init__(self, K, N):
        """Initializes the PPM

        Args:
            K (int): Number of neurons.
            N (int): Number of weights per neuron.
        """
        # Hyper-parameters
        self.K = K
        self.N = N

        # Training Parameters
        self._sigma = None
        self._x = None
        self._tau = None

        self.weights = np.random.randint(0, 2, [K, N])

    def output(self, x):
        """Produces the output parity of a given input array.

        Args:
            x (numpy.ndarray): An integer array consisting of 0s and 1s.

        Returns:
            (int). A 0 or 1 depending on the model's process.
        """
        self._sigma = np.sign(np.sum(np.bitwise_xor(x, self.weights), axis=0) - self.N / 2)
        self._sigma[np.where(self._sigma < 0)] = 0
        self._tau = np.bitwise_xor.reduce(self._sigma)

        self._sigma = self._sigma.reshape(-1, 1)

        # Create copy of input
        self._x = np.empty_like(x)
        self._x[:] = x

        return self._tau

    def update(self, tau_b):
        """Updates the PPM model using a hebbian learning rule.

        Args:
            tau_b (int): The output of the other given machine used to train the PPM.

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
        """Returns the weights as a K*N integer array.

        Returns:
            (numpy.ndarray). Integer array consisting of 0s and 1s.
        """
        return self.weights.reshape(self.K * self.N)
