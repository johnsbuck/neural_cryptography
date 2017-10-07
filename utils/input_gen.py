import numpy as np
from numpy.matlib import repmat


def generate(low, high, size):
    return np.random.choice([low, high], size=size)


def gen_tpm_inputs(K, N, one_dim=False):
    """Used to generate inputs for the TreeParityMachine (TPM).

    Args:
        K (int): Number of neurons in TPM.
        N (int): Number of weights per neuron in TPM.
        one_dim (bool): If True, will generate a [1, k*n] array. Otherwise, will generate a [k, n] array.
            (Optional: False)

    Returns:
        (numpy.ndarray). Input used to train TPM.
    """
    if one_dim:
        return generate(-1, 1, [1, K*N])
    return generate(-1, 1, [K, N])


def gen_ppm_inputs(K, N, one_dim=False):
    """Used to generate inputs for the PermutationParityMachine (PPM).

        Args:
            K (int): Number of neurons in PPM.
            N (int): Number of weights per neuron in PPM.
            one_dim (bool): If True, will generate a [1, k*n] array. Otherwise, will generate a [k, n] array.
                (Optional: False)

        Returns:
            (numpy.ndarray). Input used to train PPM.
        """
    if one_dim:
        return generate(0, 1, [1, K*N])
    return generate(0, 1, [K, N])


def gen_tpm_rep_inputs(K, N, one_dim=False, rep=True):
    """Used to generate repeated inputs for the TreeParityMachine (TPM).

    Args:
        K (int): Number of neurons in TPM (Repeats message K times)
        N (int): Message size.
        one_dim (bool): If True, will generate a [1, k*n] array. Otherwise, will generate a [k, n] array.
            (Optional: False)
        rep (bool): If True, will generate a N-length array repeated K times (dimension dependent on one_dim).
            Otherwise, will generate a [1, N]-shaped array.
            (Optional: True)


    Returns:
        (numpy.ndarray). Input used to train TPM.
    """
    x = generate(0, 1, [1, N])

    if not rep:
        return x

    if one_dim:
        return repmat(x, 1, K)
    return repmat(x, K, 1)


def gen_ppm_rep_inputs(K, N, one_dim=False, rep=True):
    """Used to generate repeated inputs for the PermutationParityMachine (PPM).

    Args:
        K (int): Number of neurons in PPM (Repeats message K times)
        N (int): Message size.
        one_dim (bool): If True, will generate a [1, k*n] array. Otherwise, will generate a [k, n] array.
            (Optional: False)
        rep (bool): If True, will generate a N-length array repeated K times (dimension dependent on one_dim).
            Otherwise, will generate a [1, N]-shaped array.
            (Optional: True)

    Returns:
        (numpy.ndarray). Input used to train PPM.
    """

    x = generate(0, 1, [1, N])

    if not rep:
        return x

    if one_dim:
        return repmat(x, 1, K)
    return repmat(x, K, 1)
