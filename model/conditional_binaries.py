import tensorflow as tf
import numpy as np
import gc
from model import net_proto
from time import gmtime, strftime
from util import eval_tools
from util import layers


def loss_regu(par_list, weight=0.005):
    single_regu = [tf.nn.l2_loss(v) for v in par_list]
    loss = tf.add_n(single_regu) * weight
    return loss


class ConditionalVairationalBinaries(object):
    def __init__(self, **kwargs):
        self.sess = kwargs.get('sess')
        self.batch_size = kwargs.get('batch_size')
        self.feature_length = kwargs.get('feature_length')
        self.code_length = kwargs.get('code_length')
        self.latent_size = kwargs.get('latent_size')
        self.label_num = kwargs.get('label_num')
        self.restore_file = kwargs.get('restore_file')
        self.img_in = tf.placeholder(tf.float32, [self.batch_size, self.feature_length])
        self.data_label = tf.placeholder(tf.float32, [self.batch_size, self.label_num])
        self.latent_feature = tf.placeholder(tf.float32, [self.batch_size, self.latent_size])
        self.eval_map = tf.placeholder(tf.float32, [1])
        self.global_step = tf.Variable(0, False, name='global_step')
        tf.summary.scalar(net_proto.PHASE_EVAL + '/MAP', tf.reduce_sum(self.eval_map))

        self.nets = self._build_graph()
        self.loss = self._build_loss()
        self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.loss, global_step=self.global_step)
        self.out_tensor = tf.to_float(tf.greater(self.nets[4], 0))

    def _build_graph(self):
        """
        To build the networks: p(z|x), q(b|x,z), q(z|x,b) and some other components
        :return:
        """
        variational_mean, variational_log_sigma = net_proto.variational_net(self.img_in, self.latent_size,
                                                                            net_proto.SUB_SCOPE_VARIATION_PZX)

        # step 1: sample the latent space to obtain z
        eps = tf.random_normal([self.batch_size, self.latent_size], stddev=0.02)
        sampled_latent_reps = variational_mean + tf.multiply(eps, variational_log_sigma)
        # sampled_latent_reps = eps

        # step 2: render z and x to q(b|z,x) and then obtain sampled b
        feature_in_qb = tf.concat([self.img_in, sampled_latent_reps], axis=1)
        code_prototype = net_proto.generative_net(feature_in_qb, self.code_length)
        code_out = tf.to_float(tf.greater(code_prototype, 0)) * 2 - 1

        # step 3: build q(z|b,x)
        feature_in_qz = tf.concat([self.img_in, code_prototype], axis=1)
        # feature_in_qz = code_prototype
        variational_mean_1, variational_log_sigma_1 = net_proto.variational_net(feature_in_qz, self.latent_size,
                                                                                net_proto.SUB_SCOPE_VARIATION_PZXB)

        tf.summary.histogram(net_proto.PHASE_TRAIN + '/code_hist', code_out)
        tf.summary.histogram(net_proto.PHASE_TRAIN + '/net_hist', code_prototype)
        tf.summary.scalar(net_proto.PHASE_TRAIN + '/code_sum', tf.reduce_mean((code_out + 1.) / 2))
        tf.summary.histogram(net_proto.PHASE_TRAIN + '/qz_mean', variational_mean_1)
        tf.summary.histogram(net_proto.PHASE_TRAIN + '/qz_var', variational_log_sigma_1)
        code_out_img = tf.expand_dims(tf.matmul(code_out, code_out, transpose_b=True), axis=0)
        code_out_img = tf.expand_dims(code_out_img, axis=-1)
        tf.summary.image(net_proto.PHASE_TRAIN + '/code_orthogonality', code_out_img)
        return variational_mean, variational_log_sigma, variational_mean_1, variational_log_sigma_1, code_prototype, sampled_latent_reps

    def _build_loss(self):
        # learning objective 1: code regression
        loss_1 = tf.nn.l2_loss(self.nets[4] - (tf.to_float(tf.greater(self.nets[4], 0)) * 2 - 1))
        tf.summary.scalar(net_proto.PHASE_TRAIN + '/losses/quantization', loss_1)

        # learning objective 2: kl-divergence KL(q(z|x,b)||p(z|x)).
        loss_2 = tf.reduce_mean(net_proto.loss_kl(self.nets[2], self.nets[3], self.nets[0], self.nets[1]))
        tf.summary.scalar(net_proto.PHASE_TRAIN + '/losses/kl', loss_2)

        # learning objective 3: orthogonality
        loss_3 = tf.nn.l2_loss(
            tf.matmul(self.nets[4], self.nets[4], transpose_b=True) - tf.eye(self.batch_size, self.batch_size))
        # loss_3 = -1 * tf.trace(tf.matmul(self.nets[4], self.nets[4], transpose_a=True)) / (self.code_length * 2) * 100
        tf.summary.scalar(net_proto.PHASE_TRAIN + '/losses/code_variance', loss_3)

        # learning objective 4: Laplacian-graph-based loss````````
        corr_matrix = net_proto.laplacian_graph(self.img_in, self.batch_size)
        norm_z = tf.nn.l2_normalize(self.nets[5], 1)
        product = tf.matmul(tf.matmul(norm_z, corr_matrix, transpose_a=True), norm_z)
        loss_4 = -tf.trace(product)
        tf.summary.scalar(net_proto.PHASE_TRAIN + '/losses/graph_loss', loss_4)
        p = tf.expand_dims(product, axis=0)
        p = tf.expand_dims(p, axis=-1)
        tf.summary.image(net_proto.PHASE_TRAIN + '/lg', p)

        # learning objective r: L2 regularization of the network
        loss_r = loss_regu(tf.trainable_variables())
        tf.summary.scalar(net_proto.PHASE_TRAIN + '/losses/regu', loss_r)
        #
        # cls = layers.fc_layer('hehe', self.nets[-2], self.label_num)
        # loss_z = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.data_label, logits=cls))

        loss_z = tf.nn.l2_loss(self.nets[-1] - self.latent_feature)
        tf.summary.scalar(net_proto.PHASE_TRAIN + '/losses/zz', loss_z)

        loss_out = 0.8 * loss_1 + loss_2 + loss_3 + loss_4 + loss_r + loss_z
        tf.summary.scalar(net_proto.PHASE_TRAIN + '/losses/total_loss', loss_out)

        return loss_out

    def train(self, dataset, log_path, save_path):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.restore_file is not None:
            self._restore()

        max_epoch = 40
        train_interval, test_interval = dataset.iter_num()

        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=net_proto.PHASE_TRAIN))
        summary_op_eval = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=net_proto.PHASE_EVAL))

        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        writer = tf.summary.FileWriter(log_path + '/' + time_string + '/')
        saver = tf.train.Saver()

        for i in range(max_epoch):
            for j in range(train_interval):
                batch_data, batch_cluster = dataset.next_batch_train()
                batch_code, summaries, this_loss, _ = self.sess.run([self.out_tensor, summary_op, self.loss, self.opt],
                                                                    feed_dict={self.img_in: batch_data,
                                                                               self.latent_feature: batch_cluster})
                step = tf.train.global_step(self.sess, self.global_step)
                writer.add_summary(summaries, global_step=step)
                dataset.apply_code(batch_code)
                print('Epoch ' + str(i) + ', Batch ' + str(j) + '(Global Step: ' + str(step) + '): ' + str(
                    this_loss))
                del batch_code, summaries, this_loss
                gc.collect()

            print('Testing...')
            for j in range(test_interval):
                batch_data = dataset.next_batch_test()
                batch_code = self._forward(batch_data)
                dataset.apply_code(batch_code, 1)
                del batch_code, batch_data
                gc.collect()

            mean_average_precision = eval_tools.eval_cls_map(dataset.code_test, dataset.code_train, dataset.label_test,
                                                             dataset.label_train)
            step = tf.train.global_step(self.sess, self.global_step)
            eval_to_write = self.sess.run(summary_op_eval,
                                          feed_dict={self.eval_map: np.asarray([mean_average_precision])})
            writer.add_summary(eval_to_write, global_step=step)
            saver.save(self.sess, save_path + '\\ymmodel', global_step=step)
            print(str(mean_average_precision))
            dataset.reshuffle()
            del eval_to_write, mean_average_precision
            gc.collect()

    def _forward(self, batch_data):
        return self.sess.run(self.out_tensor, feed_dict={self.img_in: batch_data})

    def _restore(self, restore_file=None):
        saver = tf.train.Saver()
        if restore_file is None:
            restore_file = self.restore_file
        return saver.restore(self.sess, restore_file)

    def run_extraction(self, dataset, save_path):
        train_interval, test_interval = dataset.iter_num()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.restore_file is not None:
            self._restore()

        for i in range(train_interval):
            print(str(i))
            batch_feat, _ = dataset.next_batch_train()
            batch_code = self._forward(batch_feat)
            dataset.apply_code(batch_code)

        for i in range(test_interval):
            batch_feat = dataset.next_batch_test()
            batch_code = self._forward(batch_feat)
            dataset.apply_code(batch_code, 1)

        dataset.save(save_path)
