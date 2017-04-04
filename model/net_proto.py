import tensorflow as tf
from util import layers

PHASE_TRAIN = 'train'
PHASE_EVAL = 'eval'

NAME_SCOPE_VARIATION = 'variational_net'
SUB_SCOPE_VARIATION_PZX = 'pzx'
SUB_SCOPE_VARIATION_PZXB = 'pzxb'
NAME_SCOPE_GENERATION = 'generative_net'
NAME_SCOPE_RECOGNITION = 'recognition_net'
NAME_SCOPE_CODE_LEARNING = 'code_learning'


def variational_net(input_tensor, latent_size, sub_scope):
    """
    A variational approximation of p(z) with neural network.
    :param input_tensor:
    :param latent_size: the size of the latent space
    :param sub_scope:
    :return variational_mean:
    :return variational_log_sigma:
    """
    with tf.variable_scope(NAME_SCOPE_VARIATION + '/' + sub_scope):
        fc_1 = layers.fc_relu_layer('fc1', input_tensor, 1024)
        fc_2 = layers.fc_relu_layer('fc2', fc_1, 1024)
        variational_mean = tf.sigmoid(layers.fc_layer('v_mean', fc_2, latent_size))
        variational_log_sigma = tf.sigmoid(layers.fc_layer('log_sigma', fc_2, latent_size))
    return variational_mean, variational_log_sigma


def generative_net(input_tensor, code_length):
    """
    A network to rebuild the data from latent representation.
    :param input_tensor:
    :param code_length: hashing length
    :return fc_3:

    """
    with tf.variable_scope(NAME_SCOPE_GENERATION):
        fc_1 = layers.fc_relu_layer('fc1', input_tensor, 1024)
        fc_2 = layers.fc_relu_layer('fc2', fc_1, 1024)
        fc_3 = tf.tanh(layers.fc_layer('fc3', fc_2, code_length))
    return fc_3


def loss_kl(variational_mean_1, variational_log_sigma_1, variational_mean_2=None, variational_log_sigma_2=None):
    """
    KL-Divergence of two Gaussian distributions:
        log(s2/s1) +  (s1^2 + (u1-u2)^2)/(2*s2^2) -1/2
    :param variational_mean_1:
    :param variational_log_sigma_1:
    :param variational_mean_2:
    :param variational_log_sigma_2:
    :return:
    """
    if (variational_mean_2 is None) and (variational_log_sigma_2 is None):
        l = -0.5 * tf.reduce_mean(
            1 + variational_log_sigma_1 - tf.square(variational_mean_1) - tf.exp(variational_log_sigma_1), 1)
        return l
    else:
        l1 = (variational_log_sigma_2 - variational_log_sigma_1) * 0.5
        l2 = tf.div(tf.exp(variational_log_sigma_1) + tf.square((variational_mean_1 - variational_mean_2)),
                    2 * tf.exp(variational_log_sigma_2 + 1e-6))
        return tf.reduce_mean(l1 + l2 + 0.5, 1)


def laplacian_graph(input_tensor, keep_size=10):
    """
    quantized correlation matrix
    :param input_tensor:
    :param keep_size: you guess
    :return:
    """
    # <1> distance matrix
    squared_sum = tf.reshape(tf.reduce_sum(input_tensor * input_tensor, 1), [-1, 1])
    distances = squared_sum - 2 * tf.matmul(input_tensor, input_tensor, transpose_b=True) + tf.transpose(squared_sum)

    # <2> [0, 1] correlation
    _, k_indices = tf.nn.top_k(-distances, k=keep_size, sorted=False)
    exp_dim = tf.expand_dims(tf.range(0, tf.shape(k_indices)[0]), 1)
    exp_dim = tf.tile(exp_dim, [1, keep_size])
    full_indices = tf.concat([tf.expand_dims(exp_dim, 2), tf.expand_dims(k_indices, 2)], 2)
    full_indices = tf.reshape(full_indices, [-1, 2])
    corr = tf.cast(tf.sparse_to_dense(full_indices, [100, 100], sparse_values=1., validate_indices=False),
                   dtype=tf.float32) - tf.eye(tf.shape(distances)[0], dtype=tf.float32)

    # <3> L matrix
    d_matrix = tf.diag(1 / tf.sqrt(tf.reduce_sum(corr, 1)))
    l_m = tf.matmul(tf.matmul(d_matrix, corr), d_matrix)

    return l_m
