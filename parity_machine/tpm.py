import numpy as np
import utils.learning_rules as rules


class TreeParityMachine(object):
    """Tree Parity Machine
    Used in Neural Cryptography to generate a key of size K*N via its weights. Weights consist of integers from -L to L.

    """

    def __init__(self, k, n, l):
        """

        Args:
            k (int): Number of hidden neurons in hidden layer.
            n (int): Number of weights for each neuron.
            l (int): Range of integers for weights from -l to l.
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
        """Produces the output parity of a given input array.

        Args:
            x (numpy.ndarray): An integer array consisting of -1s and 1s.

        Returns:
            (int). A 0 or 1 depending on the model's process.
        """
        self._sigma = np.sign(np.sum(x * self.weights, axis=1)).reshape(-1, 1)
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

        self.weights = rules.hebbian(self.weights, self._x, self._sigma, -self.l, self.l, self._tau, tau_b)

        # Reset training variables for next update
        self._sigma = None
        self._x = None
        self._tau = None

    def get_key(self):
        """Returns the weights as a K*N integer array.

        Returns:
            (numpy.ndarray). Integer array consisting of 0s and 1s.
        """
        return self.weights.reshape(self.k * self.n)