import numpy as np


def generate_input(k, n, one_dim=False):
    """Used to generate inputs for the TreeParityMachine (TPM).

    Args:
        k (int): Number of neurons in TPM.
        n (int): Number of weights per neuron in TPM.
        one_dim: If True, will generate a [1, k*n] array. Otherwise, will generate a [k, n] array.
            (Optional: False)

    Returns:
        (numpy.ndarray). Input used to train TPM.
    """
    if one_dim:
        return np.random.choice([1, -1], size=k*n)
    return np.random.choice([1, -1], size=[k, n])

def generate_perm_input(k, n, one_dim=False):
    """Used to generate inputs for the PermutationParityMachine (PPM).

        Args:
            k (int): Number of neurons in PPM.
            n (int): Number of weights per neuron in PPM.
            one_dim: If True, will generate a [1, k*n] array. Otherwise, will generate a [k, n] array.
                (Optional: False)

        Returns:
            (numpy.ndarray). Input used to train PPM.
        """
    if one_dim:
        return np.random.choice([1, 0], size=k*n)
    return np.random.choice([1, 0], size=[k, n])
