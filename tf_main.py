import tensorflow as tf
from tf_parity_machine import TFTreeParityMachine
from utils.input_gen import generate_input

K = 3
N = 10
L = 10

def make_input_tensor():
    return tf.subtract(tf.multiply(tf.floor(tf.multiply(tf.random_uniform([1, K*N, 1]), 2)), 2), 1)

tree = TFTreeParityMachine(K, N, L)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in xrange(1, 100):
        input_vector = make_input_tensor()
        tree.output(input_vector, sess)
    print sess.run(tree.get_key())
