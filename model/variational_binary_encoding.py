import tensorflow as tf
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
        variational_log_sigma = layers.fc_layer('LogSiagma', fc_2, latent_size)
    return variational_mean, variational_log_sigma


def generative_net(input_tensor):
    """
    A network to rebuild the data from latent representation.
    :param input_tensor:
    :return fc_3:

    """
    with tf.variable_scope(NAME_SCOPE_GENERATION):
        fc_1 = layers.fc_relu_layer('Fc1', input_tensor, 1024)
        fc_2 = layers.fc_relu_layer('Fc2', fc_1, 1024)
        fc_3 = layers.fc_layer('Fc3', fc_2, 1024)
        fc_3 = tf.sigmoid(fc_3)
    return fc_3


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


def losses(variational_mean, variational_log_sigma, generated_data, real_data, pre_labels=None, labels=None):
    assert (pre_labels is None) == (labels is None)
    # KL-Divergence of q(z|x) and p(z)
    variational_loss = -0.5 * tf.reduce_sum(
        1 + variational_log_sigma - tf.square(variational_mean) - tf.exp(variational_log_sigma), axis=1)

    # p(x|z)
    generative_loss = tf.reduce_sum()
