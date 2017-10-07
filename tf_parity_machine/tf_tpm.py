import tensorflow as tf

class TFTreeParityMachine(object):
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
        #Hyper-parameters
        self.K = K
        self.L = L
        self.N = N

        self.x = tf.placeholder(tf.float32, shape=(1, N*K, 1), name="input")
        self.sigmas = tf.layers.conv1d(inputs=self.x,
                                  filters=K,
                                  kernel_size=N,
                                  strides=N,
                                  activation=tf.sign)
        self.tau = tf.reduce_prod(self.sigmas)

    def output(self, input_vector, sess):
        """Produces the output parity of a given input array.

        Args:
            input_vector (tensorflow.Tensor): An Tensor consisting of -1s and 1s.

        Returns:
            (int). A -1 or 1 depending on the model's process.
        """
        output = sess.run(self.tau, feed_dict={self.x: input_vector.eval()})

        return output

    def update(self, tau_b, sess):
        """Updates the TPM model using a hebbian learning rule.

        Args:
            tau_b (int): The output of the other given machine used to train the PPM.

        Returns:

        """
        pass

    def get_key(self):
        """Returns the weights as a K*N integer array.

        Returns:
            (numpy.ndarray). Integer array consisting of 0s and 1s.
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
