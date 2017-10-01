import numpy as np


def hebbian(weights, x, sigma, low, high, tau, tau_b):
    """

    Args:
        weights:
        x:
        sigma:
        low:
        high:
        tau:
        tau_b:

    Returns:

    """
    return np.clip(weights + (sigma * x * _same(sigma, tau) * _same(tau, tau_b)), low, high)


def anti_hebbian(weights, x, sigma, low, high, tau, tau_b):
    """

        Args:
            weights:
            x:
            sigma:
            low:
            high:
            tau:
            tau_b:

        Returns:

        """
    return np.clip(weights - (sigma * x * _same(sigma, tau) * _same(tau, tau_b)), low, high)


def random_walk(weights, x, sigma, low, high, tau, tau_b):
    """

        Args:
            weights:
            x:
            sigma:
            low:
            high:
            tau:
            tau_b:

        Returns:

        """
    return np.clip(weights + (x * _same(sigma, tau) * _same(tau, tau_b)), low, high)


def _same(a, b):
    """

    Args:
        a:
        b:

    Returns:

    """
    return np.equal(a, b).astype(int)
