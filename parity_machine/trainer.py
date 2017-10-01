import numpy as np
from utils import generate_input, generate_perm_input


class PermutationParityTrainer(object):
    """

    """

    def __init__(self):
        pass

    def train_step(self, model_a, model_b):
        """Updates both models if output is same.

        Args:
            model_a:
            model_b:

        Returns:

        """
        if not self.hyperparams_same(model_a, model_b):
            raise ValueError("Models are incompatible. Need same hyper-parameters (K, N, L).")

        x = generate_input(model_a.k, model_a.n)

        tau_a = model_a.output(x)
        tau_b = model_b.output(x)

        if tau_a == tau_b:
            model_a.update(tau_b)
            model_b.update(tau_a)

        return tau_a == tau_b

    def train_step(self, model_a, model_b, eve):
        """

        Args:
            model_a:
            model_b:
            eve:

        Returns:

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

        return (tau_a == tau_b) and (tau_a == tau_eve)

    def train(self, model_a, model_b, total_iter=np.infty, step=None):
        """

        Args:
            model_a:
            model_b:
            total_iter:

        Returns:

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
        """

        Args:
            model_a:
            model_b:
            eve:
            total_iter:

        Returns:

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
        """

        Args:
            model_a:
            model_b:

        Returns:
            Boolean. True if the hyper-parameters are the same, False otherwise.
        """
        if model_a.k == model_b.k and model_a.n == model_b.n:
            if 'l' not in vars(model_a) and 'l' not in vars(model_b) or model_a.l == model_b.l:
                return True
        return False


class TreeParityTrainer(object):
    """

    """

    def __init__(self):
        pass

    def train_step(self, model_a, model_b):
        """Updates both models if output is same.

        Args:
            model_a:
            model_b:

        Returns:

        """
        if not self.hyperparams_same(model_a, model_b):
            raise ValueError("Models are incompatible. Need same hyper-parameters (K, N, L).")

        x = generate_input(model_a.k, model_a.n)

        tau_a = model_a.output(x)
        tau_b = model_b.output(x)

        if tau_a == tau_b:
            model_a.update(tau_b)
            model_b.update(tau_a)

        return tau_a == tau_b

    def train_step(self, model_a, model_b, eve):
        """

        Args:
            model_a:
            model_b:
            eve:

        Returns:

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

        return (tau_a == tau_b) and (tau_a == tau_eve)

    def train(self, model_a, model_b, total_iter=np.infty, step=None):
        """

        Args:
            model_a:
            model_b:
            total_iter:

        Returns:

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
        """

        Args:
            model_a:
            model_b:
            eve:
            total_iter:

        Returns:

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
        """

        Args:
            model_a:
            model_b:

        Returns:
            Boolean. True if the hyper-parameters are the same, False otherwise.
        """
        if model_a.k == model_b.k and model_a.n == model_b.n:
            if 'l' not in vars(model_a) and 'l' not in vars(model_b) or  model_a.l == model_b.l:
                return True
        return False
