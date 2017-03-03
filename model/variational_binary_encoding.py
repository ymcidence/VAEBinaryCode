import tensorflow as tf
import cmath
from util import layers

NAME_SCOPE_VARIATION = 'VariationalNet'
NAME_SCOPE_GENERATION = 'GenerativeNet'
NAME_SCOPE_RECOGNITION = 'RecognitionNet'


def variational_net(input_tensor, latent_size):
    """
    A variational approximation of p(z) with neural network.
    :param input_tensor:
    :param latent_size: the size of the latent space
    :return variational_mean:
    :return variational_log_sigma:
    """
    with tf.variable_scope(NAME_SCOPE_VARIATION):
        fc_1 = layers.fc_relu_layer('Fc1', input_tensor, 1024)
        fc_2 = layers.fc_relu_layer('Fc2', fc_1, 1024)
        variational_mean = layers.fc_layer('VMean', fc_2, latent_size)
        variational_log_sigma = layers.fc_layer('LogSigma', fc_2, latent_size)
    return variational_mean, variational_log_sigma


def generative_net(input_tensor):
    """
    A network to rebuild the data from latent representation.
    :param input_tensor:
    :return fc_3_mean:
    :return fc_3_sigma:

    """
    with tf.variable_scope(NAME_SCOPE_GENERATION):
        fc_1 = layers.fc_relu_layer('Fc1', input_tensor, 1024)
        fc_2 = layers.fc_relu_layer('Fc2', fc_1, 1024)
        fc_3_mean = layers.fc_layer('Fc3Mean', fc_2, 1024)
        fc_3_sigma = layers.fc_layer('Fc3Sigma', fc_2, 1024)
    return fc_3_mean, fc_3_sigma


def recognition_net(input_tensor, label_size):
    """
    A recognition network as an auxiliary to the learning process.
    :param input_tensor:
    :param label_size: the number of all classes involved
    :return prob:
    """
    with tf.variable_scope(NAME_SCOPE_RECOGNITION):
        fc_1 = layers.fc_relu_layer('Fc1', input_tensor, 512)
        fc_2 = layers.fc_relu_layer('Fc2', fc_1, 512)
        prob = layers.fc_layer('Prob', fc_2, label_size)
    return prob


def loss_kl(variational_mean, variational_log_sigma):
    """
    KL-Divergence of q(z|x) and p(z)
    :param variational_mean:
    :param variational_log_sigma:
    :return:
    """
    variational_loss = -0.5 * tf.reduce_sum(
        1 + variational_log_sigma - tf.square(variational_mean) - tf.exp(variational_log_sigma), axis=1)
    return variational_loss


def loss_px(generated_mean, generated_sigma, real_data):
    """
    p(x|z)
    :param generated_mean:
    :param generated_sigma:
    :param real_data:
    :return:
    """
    c = -0.5 * cmath.log(2 * cmath.pi)
    generative_loss = tf.square(real_data - generated_mean) / (2. * tf.exp(generated_sigma)) - c + generated_sigma / 2.
    return generative_loss


class BinaryEncodingVae(object):
    def __init__(self, code_length, sess=tf.Session()):
        self.sess = sess
        self.code_length = code_length
        self.data_feature = tf.placeholder(tf.float32, [None, None])
        self.matrix_r = tf.Variable(initial_value=tf.eye(code_length, code_length), trainable=False)
