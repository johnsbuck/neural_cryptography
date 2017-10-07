import numpy as np
from parity_machine import TreeParityMachine
from utils import gen_tpm_inputs


def train_step(model_a, model_b):
    """Updates model a and model b if they have the same tau given the random input.

    Args:
        model_a (TreeParityMachine): Model used to sync weights with Model B.
        model_b (TreeParityMachine): Model used to sync weights with Model A.

    Returns:
        (bool). If True, then Model A and B updated.
    """
    if not hyperparams_same(model_a, model_b):
        raise ValueError("Models are incompatible. Need same hyper-parameters (K, N, L).")

    x = gen_tpm_inputs(model_a.k, model_a.n)

    tau_a = model_a.output(x)
    tau_b = model_b.output(x)

    if tau_a == tau_b:
        model_a.update(tau_b)
        model_b.update(tau_a)

    return tau_a == tau_b


def train_step(model_a, model_b, eve):
    """Updates model a and model b if they have the same tau given the random input.
    Will also update the snooper model, eve, if matches with model a and model b.

    Args:
        model_a (TreeParityMachine): Model used to sync weights with Model B.
        model_b (TreeParityMachine): Model used to sync weights with Model A.
        eve (TreeParityMachine): Snooper model attempting to sync with Model A and B.

    Returns:
        (tuple). If first element is True, then Model A and B updated.
        If the second element is True then Eve also updated.
    """
    if not hyperparams_same(model_a, model_b) or not hyperparams_same(model_a, eve):
        raise ValueError("Models are incompatible. Need same hyper-parameters (K, N, L).")

    K, N, L = model_a.get_hyper_params()
    x = gen_tpm_inputs(K, N)

    tau_a = model_a.output(x)
    tau_b = model_b.output(x)
    tau_eve = eve.output(x)

    if tau_a == tau_b:
        model_a.update(tau_b)
        model_b.update(tau_a)

        if tau_a == tau_eve:
            eve.update(tau_a)

    return (tau_a == tau_b), (tau_a == tau_b) and (tau_a == tau_eve)


def train(model_a, model_b, total_iter=np.infty, print_step=None):
    """Runs through several training steps with model A and B, attempting to have a closer match.

    Args:
        model_a (TreeParityMachine): Model used to sync weights with Model B.
        model_b (TreeParityMachine): Model used to sync weights with Model A.
        total_iter (int): The total number of training steps to run or until matching.
            Can be set to np.infty, which will run the training until Model A matches B.
            (Optional: np.infty)
        print_step (int): Prints training information every print_step. If None, won't print anything.
            (i.e. print_step = 100 would print information every 100 steps)
            (Optional: None)

    Returns:
        (list).
        trained (bool): If True, Model A and B are synced.
        n_iter (int): Number iterations it took to train Model A and B.
        progress (list): Consists of the matching percentage between Model A and B each iteration.
    """
    trained = False
    n_iter = 0
    progress = []
    while total_iter > 0 or not trained:
        progress.append(np.equal(model_a.get_key() == model_b.get_key(), True).sum() / np.prod(model_a.shape))

        if np.array_equal(model_a.get_key(), model_b.get_key()):
            trained = True
            break

        if print_step is not None and ((n_iter + 1) % print_step) == 0:
            print "Step:", n_iter + 1
            print "Percent Match (A & B):", progress[-1][0]
            print "Percent Match (A & Eve):", progress[-1][1]
            print "Percent Match (B & Eve):", progress[-1][2]
            print ""

        train_step(model_a, model_b)
        n_iter += 1
    return [trained, n_iter, progress]


def train(model_a, model_b, eve, total_iter=np.infty, print_step=None):
    """Runs through several training steps with model A and B, attempting to have a closer match.

    Args:
        model_a (TreeParityMachine): Model used to sync weights with Model B.
        model_b (TreeParityMachine): Model used to sync weights with Model A.
        eve (TreeParityMachine): Snooper model attempting to sync with Model A and B.
        total_iter (int): The total number of training steps to run or until matching.
            Can be set to np.infty, which will run the training until Model A matches B.
            (Optional: np.infty)
        print_step (int): Prints training information every print_step. If None, won't print anything.
            (i.e. print_step = 100 would print information every 100 steps)
            (Optional: None)

    Returns:
        (list).
        trained (bool): If True, Model A and B are synced.
        n_iter (int): Number iterations it took to train Model A and B.
        progress (list): Consists of the matching percentage between Model A, B, and Eve each iteration.
    """
    trained = False
    n_iter = 0
    progress = []
    while total_iter > 0 or not trained:
        progress.append([np.equal(model_a.get_key(), model_b.get_key()).sum() *
                         1. / np.prod(model_a.get_key().shape),
                         np.equal(model_a.get_key(), eve.get_key()).sum() *
                         1. / np.prod(model_a.get_key().shape),
                         np.equal(model_b.get_key(), eve.get_key()).sum() *
                         1. / np.prod(model_a.get_key().shape)])

        if np.array_equal(model_a.get_key(), model_b.get_key()):
            trained = True
            break

        if print_step is not None and ((n_iter + 1) % print_step) == 0:
            print "Step:", n_iter + 1
            print "Percent Match (A & B):", progress[-1][0]
            print "Percent Match (A & Eve):", progress[-1][1]
            print "Percent Match (B & Eve):", progress[-1][2]
            print ""

        train_step(model_a, model_b, eve)
        n_iter += 1
    return [trained, n_iter, progress]


def hyperparams_same(model_a, model_b):
    """Confirms that two models have the same hyper-parameters for their models.

    Args:
        model_a (TreeParityMachine): Compared with model_b.
        model_b (TreeParityMachine): Compared with model_a.

    Returns:
        Boolean. True if the hyper-parameters are the same, False otherwise.
    """
    if model_a.get_hyper_params() == model_b.get_hyper_params():
        return True
    return False
