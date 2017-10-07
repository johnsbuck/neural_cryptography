import numpy as np
import utils.learning_rules as rules


class TreeParityMachine(object):
    """Tree Parity Machine
    Used in Neural Cryptography to generate a key of size K*N via its weights. Weights consist of integers from -L to L.

    """

    def __init__(self, K, N, L):
        """

        Args:
            K (int): Number of hidden neurons in hidden layer.
            N (int): Number of weights for each neuron.
            L (int): Range of integers for weights from -L to L.
        """
        # Hyper-parameters
        self._K = K
        self._N = N
        self._L = L

        # Training Parameters
        self._sigma = None
        self._x = None
        self._tau = None

        self._weights = np.random.randint(-L, L + 1, [K, N])

    def get_hyper_params(self):
        """

        Returns:
            (List<int>) Hyper-parameters defined in initialization
        """
        return [self._K, self._N, self._L]

    def output(self, x):
        """Produces the output parity of a given input array.

        Args:
            x (numpy.ndarray): An integer array consisting of -1s and 1s.

        Returns:
            (int). A 0 or 1 depending on the model's process.
        """
        self._sigma = np.sign(np.sum(x * self._weights, axis=1)).reshape(-1, 1)
        self._sigma[np.where(self._sigma == 0)] -= 1
        self._tau = np.prod(self._sigma)

        # Create copy of input
        self._x = np.empty_like(x)
        self._x[:] = x

        return self._tau

    def update(self, tau_b):
        """Updates the TPM model using a hebbian learning rule.

        Args:
            tau_b (int): The output of the other given machine used to train the PPM.

        Returns:

        """
        if None is self._sigma or None is self._x or None is self._tau:
            raise ValueError("Did not obtain output for training. Training parameters undefined.")

        self._weights = rules.hebbian(self._weights, self._x, self._sigma, -self._L, self._L, self._tau, tau_b)

        # Reset training variables for next update
        self._sigma = None
        self._x = None
        self._tau = None

    def get_key(self):
        """Returns the weights as a K*N integer array.

        Returns:
            (numpy.ndarray). Integer array consisting of 0s and 1s.
        """
        return self._weights.reshape(self._K * self._N)
