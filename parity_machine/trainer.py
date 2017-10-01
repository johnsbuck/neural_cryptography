import numpy as np
from utils import generate_input, generate_perm_input


class PermutationParityTrainer(object):
    """Permutation Parity Trainer
    Used to train two Permutation Parity Machines (PPM) to sync weights, as well as involving a third snooper model.

    """

    def __init__(self):
        pass

    def train_step(self, model_a, model_b):
        """Updates model a and model b if they have the same tau given the random input.

        Args:
            model_a (PermutationParityMachine): Model used to sync weights with Model B.
            model_b (PermutationParityMachine): Model used to sync weights with Model A.

        Returns:
            (bool). If True, then Model A and B updated.
        """
        if not self.hyperparams_same(model_a, model_b):
            raise ValueError("Models are incompatible. Need same hyper-parameters (K, N, L).")

        x = generate_perm_input(model_a.k, model_a.n)

        tau_a = model_a.output(x)
        tau_b = model_b.output(x)

        if tau_a == tau_b:
            model_a.update(tau_b)
            model_b.update(tau_a)

        return tau_a == tau_b

    def train_step(self, model_a, model_b, eve):
        """Updates model a and model b if they have the same tau given the random input.
        Will also update the snooper model, eve, if matches with model a and model b.

        Args:
            model_a (PermutationParityMachine): Model used to sync weights with Model B.
            model_b (PermutationParityMachine): Model used to sync weights with Model A.
            eve (PermutationParityMachine): Snooper model attempting to sync with Model A and B.

        Returns:
            (tuple). If first element is True, then Model A and B updated.
            If the second element is True then Eve also updated.
        """
        if not self.hyperparams_same(model_a, model_b) or not self.hyperparams_same(model_a, eve):
            raise ValueError("Models are incompatible. Need same hyper-parameters (K, N).")

        x = generate_perm_input(model_a.k, model_a.n)

        tau_a = model_a.output(x)
        tau_b = model_b.output(x)
        tau_eve = eve.output(x)

        if tau_a == tau_b:
            model_a.update(tau_b)
            model_b.update(tau_a)

            if tau_a == tau_eve:
                eve.update(tau_a)

        return (tau_a == tau_b), (tau_a == tau_b) and (tau_a == tau_eve)

    def train(self, model_a, model_b, total_iter=np.infty, print_step=None):
        """Runs through several training steps with model A and B, attempting to have a closer match.

        Args:
            model_a (PermutationParityMachine): Model used to sync weights with Model B.
            model_b (PermutationParityMachine): Model used to sync weights with Model A.
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
            progress.append(np.equal(model_a.weights == model_b.weights, True).sum() / np.prod(model_a.shape))

            if np.array_equal(model_a.weights, model_b.weights):
                trained = True
                break

            if print_step is not None and ((n_iter + 1) % print_step) == 0:
                print "Step:", n_iter + 1
                print "Percent Match (A & B):", progress[-1][0]
                print ""

            self.train_step(model_a, model_b)
            n_iter += 1
        return [trained, n_iter, progress]

    def train(self, model_a, model_b, eve, total_iter=np.infty, print_step=None):
        """Runs through several training steps with model A and B, attempting to have a closer match.

        Args:
            model_a (PermutationParityMachine): Model used to sync weights with Model B.
            model_b (PermutationParityMachine): Model used to sync weights with Model A.
            eve (PermutationParityMachine): Model used to snoop on Models A and B.
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
            progress.append([np.equal(model_a.weights, model_b.weights).sum() * 1. / np.prod(model_a.weights.shape),
                             np.equal(model_a.weights, eve.weights).sum() * 1. / np.prod(model_a.weights.shape),
                             np.equal(model_b.weights, eve.weights).sum() * 1. / np.prod(model_a.weights.shape)])

            if np.array_equal(model_a.weights, model_b.weights):
                trained = True
                break

            if print_step is not None and ((n_iter + 1) % print_step) == 0:
                print "Step:", n_iter + 1
                print "Percent Match (A & B):", progress[-1][0]
                print "Percent Match (A & Eve):", progress[-1][1]
                print "Percent Match (B & Eve):", progress[-1][2]
                print ""

            self.train_step(model_a, model_b, eve)
            n_iter += 1
        return [trained, n_iter, progress]

    @staticmethod
    def hyperparams_same(model_a, model_b):
        """Confirms that two models have the same hyper-parameters for their models.

        Args:
            model_a (PermutationParityMachine): Compared with model_b.
            model_b (PermutationParityMachine): Compared with model_a.

        Returns:
            Boolean. True if the hyper-parameters are the same, False otherwise.
        """
        if model_a.k == model_b.k and model_a.n == model_b.n:
                return True
        return False


class TreeParityTrainer(object):
    """Tree Parity Trainer
    Used to train two Tree Parity Machines (TPM) to sync weights, as well as involving a third snooper model.

    """

    def __init__(self):
        pass

    def train_step(self, model_a, model_b):
        """Updates model a and model b if they have the same tau given the random input.

        Args:
            model_a (TreeParityMachine): Model used to sync weights with Model B.
            model_b (TreeParityMachine): Model used to sync weights with Model A.

        Returns:
            (bool). If True, then Model A and B updated.
        """
        if not self.hyperparams_same(model_a, model_b):
            raise ValueError("Models are incompatible. Need same hyper-parameters (K, N, L).")

        x = generate_input(model_a.k, model_a.n)

        tau_a = model_a.output(x)
        tau_b = model_b.output(x)

        if tau_a == tau_b:
            model_a.update(tau_b)
            model_b.update(tau_a)

        return (tau_a == tau_b), (tau_a == tau_b) and (tau_a == tau_eve)

    def train_step(self, model_a, model_b, eve):
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
        if not self.hyperparams_same(model_a, model_b) or not self.hyperparams_same(model_a, eve):
            raise ValueError("Models are incompatible. Need same hyper-parameters (K, N, L).")

        x = generate_input(model_a.k, model_a.n)

        tau_a = model_a.output(x)
        tau_b = model_b.output(x)
        tau_eve = eve.output(x)

        if tau_a == tau_b:
            model_a.update(tau_b)
            model_b.update(tau_a)

            if tau_a == tau_eve:
                eve.update(tau_a)

        return (tau_a == tau_b), (tau_a == tau_b) and (tau_a == tau_eve)

    def train(self, model_a, model_b, total_iter=np.infty, step=None):
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
            progress.append(np.equal(model_a.weights == model_b.weights, True).sum() / np.prod(model_a.shape))

            if np.array_equal(model_a.weights, model_b.weights):
                trained = True
                break

            if step is not None and ((n_iter + 1) % step) == 0:
                print "Step:", n_iter + 1
                print "Percent Match (A & B):", progress[-1][0]
                print "Percent Match (A & Eve):", progress[-1][1]
                print "Percent Match (B & Eve):", progress[-1][2]
                print ""

            self.train_step(model_a, model_b)
            n_iter += 1
        return [trained, n_iter, progress]

    def train(self, model_a, model_b, eve, total_iter=np.infty, step=None):
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
            progress.append([np.equal(model_a.weights, model_b.weights).sum() * 1. / np.prod(model_a.weights.shape),
                             np.equal(model_a.weights, eve.weights).sum() * 1. / np.prod(model_a.weights.shape),
                             np.equal(model_b.weights, eve.weights).sum() * 1. / np.prod(model_a.weights.shape)])

            if np.array_equal(model_a.weights, model_b.weights):
                trained = True
                break

            if step is not None and ((n_iter + 1) % step) == 0:
                print "Step:", n_iter + 1
                print "Percent Match (A & B):", progress[-1][0]
                print "Percent Match (A & Eve):", progress[-1][1]
                print "Percent Match (B & Eve):", progress[-1][2]
                print ""

            self.train_step(model_a, model_b, eve)
            n_iter += 1
        return [trained, n_iter, progress]

    @staticmethod
    def hyperparams_same(model_a, model_b):
        """Confirms that two models have the same hyper-parameters for their models.

        Args:
            model_a (TreeParityMachine): Compared with model_b.
            model_b (TreeParityMachine): Compared with model_a.

        Returns:
            Boolean. True if the hyper-parameters are the same, False otherwise.
        """
        if model_a.k == model_b.k and model_a.n == model_b.n and model_a.l == model_b.l:
                return True
        return False
