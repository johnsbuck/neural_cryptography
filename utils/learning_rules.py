import numpy as np


def hebbian(weights, x, sigma, low, high, tau, tau_b):
    """Hebbian learning rule used to train Parity Machines.

    Args:
        weights (numpy.ndarray): Parity Machine weights that are updated.
        x (numpy.ndarray): Input array used to update weights.
        sigma (numpy.ndarray): Result of calculating previous output (tau) with weights and x.
        low (int): Minimum value of weights.
        high (int): Maximum value of weights.
        tau (int): Output of parity machine based on weights, x, and sigma.
        tau_b (int): Output of other parity machine that is attempting to sync with given parity machine weights.

    Returns:
        (numpy.ndarray) Updated weights.
    """
    return np.clip(weights + (sigma * x * same(sigma, tau) * same(tau, tau_b)), low, high)


def anti_hebbian(weights, x, sigma, low, high, tau, tau_b):
    """Anti-Hebbian learning rule used to train Parity Machines.

    Args:
        weights (numpy.ndarray): Parity Machine weights that are updated.
        x (numpy.ndarray): Input array used to update weights.
        sigma (numpy.ndarray): Result of calculating previous output (tau) with weights and x.
        low (int): Minimum value of weights.
        high (int): Maximum value of weights.
        tau (int): Output of parity machine based on weights, x, and sigma.
        tau_b (int): Output of other parity machine that is attempting to sync with given parity machine weights.

    Returns:
        (numpy.ndarray) Updated weights.
    """
    return np.clip(weights - (sigma * x * same(sigma, tau) * same(tau, tau_b)), low, high)


def random_walk(weights, x, sigma, low, high, tau, tau_b):
    """Random Walk learning rule used to train Parity Machines.

    Args:
        weights (numpy.ndarray): Parity Machine weights that are updated.
        x (numpy.ndarray): Input array used to update weights.
        sigma (numpy.ndarray): Result of calculating previous output (tau) with weights and x.
        low (int): Minimum value of weights.
        high (int): Maximum value of weights.
        tau (int): Output of parity machine based on weights, x, and sigma.
        tau_b (int): Output of other parity machine that is attempting to sync with given parity machine weights.

    Returns:
        (numpy.ndarray) Updated weights.
    """
    return np.clip(weights + (x * same(sigma, tau) * same(tau, tau_b)), low, high)


def same(a, b):
    """Compares two values. If the same, returns 1, otherwise 0.
    Typical used with two integers (tau) or a numpy.ndarray and an integer (sigma and tau).

    Args:
        a (numpy.ndarray): Compares element-wise with an integer value or compares with an equivalent-shaped array, b.
        b (numpy.ndarray): Compares element-wise with an integer value or compares with an equivalent-shaped array, a.

    Returns:
        1 if equal, 0 otherwise.
    """
    return np.equal(a, b).astype(int)
