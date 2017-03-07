import tensorflow as tf
import cmath
from time import gmtime, strftime
from util import layers

NAME_SCOPE_VARIATION = 'VariationalNet'
SUB_SCOPE_VARIATION_PZX = 'Pzx'
SUB_SCOPE_VARIATION_PZXB = 'Pzxb'
NAME_SCOPE_GENERATION = 'GenerativeNet'
NAME_SCOPE_RECOGNITION = 'RecognitionNet'
NAME_SCOPE_CODE_LEARNING = 'CodeLearning'

PATH_SUMMARY = 'E:\\WorkSpace\\WorkSpace\\TrainingLogs'
PATH_SNAPSHOT = 'E:\\WorkSpace\\WorkSpace\\SavedModels\\BVAE'


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
        fc_1 = layers.fc_relu_layer('Fc1', input_tensor, 512)
        fc_2 = layers.fc_relu_layer('Fc2', fc_1, 512)
        variational_mean = tf.sigmoid(layers.fc_layer('VMean', fc_2, latent_size))
        variational_log_sigma = tf.sigmoid(layers.fc_layer('LogSigma', fc_2, latent_size))
    return variational_mean, variational_log_sigma


def generative_net(input_tensor, code_length):
    """
    A network to rebuild the data from latent representation.
    :param input_tensor:
    :param code_length: hashing length
    :return fc_3:

    """
    with tf.variable_scope(NAME_SCOPE_GENERATION):
        fc_1 = layers.fc_relu_layer('Fc1', input_tensor, 512)
        fc_2 = layers.fc_relu_layer('Fc2', fc_1, 512)
        fc_3 = layers.fc_layer('Fc3', fc_2, code_length)
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


def loss_kl(variational_mean_1, variational_log_sigma_1, variational_mean_2, variational_log_sigma_2):
    """
    KL-Divergence of two Gaussian distributions:
        log(s2/s1) +  (s1^2 + (u1-u2)^2)/(2*s2^2) -1/2
    :param variational_mean_1:
    :param variational_log_sigma_1:
    :param variational_mean_2:
    :param variational_log_sigma_2:
    :return:
    """
    l1 = (variational_log_sigma_2 - variational_log_sigma_1) * 0.5
    l2 = tf.div(tf.exp(variational_log_sigma_1) + tf.square((variational_mean_1 - variational_mean_2)),
                2 * tf.exp(variational_log_sigma_2 + 1e-6))
    return l1 + l2 + 0.5


def loss_px(generated_mean, generated_log_sigma, real_data):
    """
    p(x|z), not used here
    :param generated_mean:
    :param generated_log_sigma:
    :param real_data:
    :return:
    """
    c = -0.5 * cmath.log(2 * cmath.pi)
    generative_loss = tf.square(real_data - generated_mean) / (
        2. * tf.exp(generated_log_sigma)) - c + generated_log_sigma / 2.
    return generative_loss


def binary_code_learning_process(matrix_f, code_length, iteration=3):
    """
    Binary code optimization process
    :param matrix_f:
    :param code_length:
    :param iteration:
    :return:
    """
    def loop_body(f, r, b, i):
        b = tf.sign(tf.matmul(r, f))
        a, _, c = tf.svd(tf.matmul(b, f, transpose_b=True))
        r = tf.matmul(a, c)
        return f, r, b, tf.add(i, 1)

    def loop_cond(f, r, b, i):
        return tf.greater(i, tf.constant(iteration))

    with tf.variable_scope(NAME_SCOPE_CODE_LEARNING, reuse=True):
        bins = tf.cast(tf.greater(tf.random_normal([code_length, code_length]), 0), tf.float32)
        iter_count = tf.constant(0)
        orthogonal_rotation = tf.get_variable('MatrixR')
        _, value_r, value_b, _ = tf.while_loop(loop_cond, loop_body, [matrix_f, orthogonal_rotation, bins, iter_count],
                                               back_prop=False)
        assign_r = tf.assign(orthogonal_rotation, value_r)

    return assign_r, value_r, value_b


class BinaryEncodingCVAE(object):
    def __init__(self, code_length, latent_size=512, feature_length=1024, label_num=60, sess=tf.Session(),
                 restore_file=None, batch_reader=None):
        self.sess = sess
        self.code_length = code_length
        self.latent_size = latent_size
        self.data_feature = tf.placeholder(tf.float32, [None, feature_length])
        self.data_label = tf.placeholder(tf.float32, [None, label_num])
        self.restore_file = restore_file
        self.batch_reader = batch_reader
        with tf.variable_scope(NAME_SCOPE_CODE_LEARNING):
            self.matrix_r = tf.Variable(initial_value=tf.eye(code_length, code_length), name='MatrixR', trainable=False)
            self.matrix_b = tf.Variable(initial_value=tf.random_normal(code_length, code_length), name='MatrixB',
                                        trainable=False)

        tf.summary.image(NAME_SCOPE_CODE_LEARNING + '/Orthogonality',
                         tf.matmul(self.matrix_r, self.matrix_r, transpose_a=True))

        self.nets = self._build_graph()
        self.loss = self._build_loss()
        self.opt = self._get_optimizer()
        self.g_step = tf.Variable(0, trainable=False)

    def _build_graph(self):
        """
        To build the networks: p(z|x), q(b|x,z), q(z|x,b) and some other components
        :return:
        """
        variational_mean, variational_log_sigma = variational_net(self.data_feature, self.latent_size,
                                                                  SUB_SCOPE_VARIATION_PZX)
        batch_size = self.data_feature.get_shape().as_list()[0]

        # step 1: sample the latent space to obtain z
        eps = tf.random_normal([batch_size, self.code_length], stddev=0.5)
        sampled_latent_reps = variational_mean + tf.multiply(eps, variational_log_sigma)

        # step 2: render z and x to q(b|z,x) and then obtain sampled b
        feature_in_qb = tf.concat([self.data_feature, sampled_latent_reps], axis=1)
        code_prototype = generative_net(feature_in_qb, self.code_length)

        # step 3: build q(z|b,x)
        feature_in_qz = tf.concat([self.data_feature, code_prototype], axis=1)
        variational_mean_1, variational_log_sigma_1 = variational_net(feature_in_qz, self.latent_size,
                                                                      SUB_SCOPE_VARIATION_PZXB)

        code_out = tf.sign(tf.matmul(self.matrix_r, code_prototype, transpose_b=True))
        tf.summary.scalar(NAME_SCOPE_GENERATION + '/CodeSum',
                          tf.reduce_mean(code_out) / (batch_size * self.code_length))

        return variational_mean, variational_log_sigma, variational_mean_1, variational_log_sigma_1, code_prototype

    def _build_loss(self):
        assigner, rotation, codes = binary_code_learning_process(tf.transpose(self.nets[4]), self.code_length)
        with tf.control_dependencies([assigner]):
            # learning objective 1: code regression
            loss_1 = tf.nn.l2_loss(tf.matmul(rotation, codes, transpose_b=True))
            tf.summary.scalar('Losses/Loss1', loss_1)

            # learning objective 2: kl-divergence between q(z|x,b) and p(z|x).
            # Note that q(z|x,b) should be placed before p(z|x)
            loss_2 = tf.reduce_sum(loss_kl(self.nets[2], self.nets[3], self.nets[0], self.nets[1]))
            tf.summary.scalar('Losses/Loss2', loss_2)

            loss_out = loss_1 + loss_2
            tf.summary.scalar('Losses/TotalLoss', loss_out)
        return loss_out

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, global_step=self.g_step)

    def _restore(self, restore_file=None):
        saver = tf.train.Saver()
        if restore_file is None:
            restore_file = self.restore_file
        return saver.restore(self.sess, restore_file)

    def run_training(self, max_iter=100000, summary_path=PATH_SUMMARY, snapshot_path=PATH_SNAPSHOT, batch_reader=None):
        if batch_reader is None:
            batch_reader = self.batch_reader
        assert batch_reader is not None

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.restore_file is not None:
            self._restore()
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        writer = tf.summary.FileWriter(summary_path + '/' + time_string + '/')
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        for i in range(max_iter):
            this_batch = batch_reader(i)
            batch_feature = this_batch.get('batch_feat')
            step_count, this_loss, summaries, _ = self.sess.run([self.g_step, self.loss, summary_op, self.opt],
                                                                feed_dict={self.data_feature: batch_feature})
            print('Iteration ' + str(i) + '(Global Step: ' + str(step_count) + '): ' + str(this_loss))
            writer.add_summary(summaries, global_step=step_count)

            if (i + 1) % 5000 == 0:
                saver.save(self.sess, snapshot_path + '/YMModel', global_step=step_count)

    def forward(self, batch_data):
        to_run = tf.matmul(self.matrix_r, self.nets[4], transpose_b=True)
        to_run = tf.sign(to_run)
        codes = self.sess.run(to_run, feed_dict={self.data_feature: batch_data.get('batch_feat')})
        return codes


if __name__ == '__main__':
    from util.data_reader import reader_feat_only

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    this_session = tf.Session(config=config)
    model = BinaryEncodingCVAE(code_length=32, feature_length=4096, label_num=397, sess=this_session,
                               batch_reader=reader_feat_only)

    model.run_training()
