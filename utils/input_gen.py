import numpy as np


def generate_input(k, n, one_dim=False):
    if one_dim:
        return np.random.choice([1, -1], size=k*n)
    return np.random.choice([1, -1], size=[k, n])

def generate_perm_input(k, n, one_dim=False):
    if one_dim:
        return np.random.choice([1, 0], size=k*n)
    return np.random.choice([1, 0], size=[k, n])
