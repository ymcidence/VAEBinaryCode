import tensorflow as tf
import cmath
from util import layers

NAME_SCOPE_VARIATION = 'VariationalNet'
NAME_SCOPE_GENERATION = 'GenerativeNet'
NAME_SCOPE_RECOGNITION = 'RecognitionNet'
NAME_SCOPE_CODE_LEARNING = 'CodeLearning'


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
        fc_3_sigma = layers.fc_layer('Fc3LogSigma', fc_2, 1024)
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


def loss_px(generated_mean, generated_log_sigma, real_data):
    """
    p(x|z)
    :param generated_mean:
    :param generated_log_sigma:
    :param real_data:
    :return:
    """
    c = -0.5 * cmath.log(2 * cmath.pi)
    generative_loss = tf.square(real_data - generated_mean) / (
        2. * tf.exp(generated_log_sigma)) - c + generated_log_sigma / 2.
    return generative_loss


def binary_code_learning_process(matrix_f, orthogonal_rotation, aux_binaries, code_length, iteration=3):
    def loop_body(f, r, b):
        a, _, c = tf.svd(tf.matmul(b, f, transpose_b=True))
        r = tf.matmul(a, c)
        b = tf.sign(tf.matmul(r, f))

        with tf.variable_scope(NAME_SCOPE_CODE_LEARNING, reuse=True):
            bins = tf.cast(tf.greater(tf.random_normal([code_length, code_length]), 0), tf.float32)
            assign_b = tf.assign(tf.get_variable('MatrixB'), bins)

    with tf.control_dependencies([assign_b]):

    return 0


class BinaryEncodingVae(object):
    def __init__(self, code_length, sess=tf.Session(), feature_length=1024, label_num=60):
        self.sess = sess
        self.code_length = code_length
        self.data_feature = tf.placeholder(tf.float32, [None, feature_length])
        self.data_label = tf.placeholder(tf.float32, [None, label_num])
        with tf.variable_scope(NAME_SCOPE_CODE_LEARNING):
            self.matrix_r = tf.Variable(initial_value=tf.eye(code_length, code_length), name='MatrixR', trainable=False)
            self.matrix_b = tf.Variable(initial_value=tf.random_normal(code_length, code_length), name='MatrixB',
                                        trainable=False)

    def _build_graph(self):
        variational_mean, variational_log_sigma = variational_net(self.data_feature, self.code_length)
        batch_size = self.data_feature.get_shape().as_list()[0]
        # sample the reparameterized latent distribution with a set of random variable 'epsilon'.
        eps = tf.random_normal([batch_size, self.code_length], stddev=0.5)
        sampled_latent_reps = variational_mean + tf.multiply(eps, variational_log_sigma)

        return 0
