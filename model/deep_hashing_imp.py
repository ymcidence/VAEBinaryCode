import tensorflow as tf
import numpy as np
import gc
from util import layers
from util import eval_tools

'''
This is the implementation of the CVPR15 paper 'Deep hashing for compact binary codes learning'
'''
NAMESCOPE_TRAIN = 'Train'
NAMESCOPE_TEST = 'Test'


class DeepHashing(object):
    def __init__(self, code_length, input_length, batch_size, sess=tf.Session()):
        self.code_length = code_length
        self.input_length = input_length
        self.batch_size = batch_size
        self.img_in = tf.placeholder(tf.float32, [batch_size, input_length])
        self.eval_map = tf.placeholder(tf.float32, [1])
        self.sess = sess
        self.net, self.matrix_w = self._build_net()
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.loss = self._build_loss()
        self.opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, self.global_step)
        self.out_tensor = tf.to_float(tf.greater(self.net, 0))
        tf.summary.scalar(NAMESCOPE_TRAIN + '/Sum', tf.reduce_sum(self.out_tensor))
        tf.summary.scalar(NAMESCOPE_TEST + '/MAP', tf.reduce_sum(self.eval_map))

    def _build_net(self):
        fc_1 = layers.fc_layer('Fc1', self.img_in, 1024)
        fc_1 = tf.tanh(fc_1)
        fc_2 = layers.fc_layer('Fc2', fc_1, 1024)
        fc_2 = tf.tanh(fc_2)

        with tf.variable_scope('Fc3'):
            fc_3_weights = tf.get_variable('weights', [1024, self.code_length],
                                           initializer=tf.random_normal_initializer(stddev=0.01))
            fc_3_biases = tf.get_variable('biases', self.code_length, initializer=tf.constant_initializer(0.))
            fc_3 = tf.nn.xw_plus_b(fc_2, fc_3_weights, fc_3_biases)
            fc_3 = tf.tanh(fc_3)

        return fc_3, fc_3_weights

    def _build_loss(self):
        """
        The loss function contains 4 terms, with weights 1, 100, 0.001 and 0.001 respectively.
        :return: total loss
        """
        quantized = tf.to_float(tf.greater(self.net, 0)) * 2 - 1.
        loss_1 = tf.nn.l2_loss(self.net - quantized)
        tf.summary.scalar(NAMESCOPE_TRAIN + '/Loss1', loss_1)

        loss_2 = -1 * tf.trace(tf.matmul(self.net, self.net, transpose_b=True)) / (self.batch_size * 2)
        tf.summary.scalar(NAMESCOPE_TRAIN + '/Loss2', loss_2)

        trainable_list = tf.trainable_variables()
        orth = [tf.nn.l2_loss(tf.matmul(v, v, transpose_a=True) - tf.eye(v.get_shape().as_list()[1])) for v in
                trainable_list if v.name.find('weights') >= 0]
        loss_3 = tf.add_n(orth)
        tf.summary.scalar(NAMESCOPE_TRAIN + '/Loss3', loss_3)

        loss_4 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_list])
        tf.summary.scalar(NAMESCOPE_TRAIN + '/Loss4', loss_4)

        loss = loss_1 + 100 * loss_2 + 0.1 * loss_3 + 0.1 * loss_4
        tf.summary.scalar(NAMESCOPE_TRAIN + '/Loss', loss)

        return loss

    def train(self, dataset, log_path, save_path):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        max_epoch = 40
        train_interval, test_interval = dataset.iter_num()

        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=NAMESCOPE_TRAIN))
        summary_op_eval = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=NAMESCOPE_TEST))

        writer = tf.summary.FileWriter(log_path + '/')
        saver = tf.train.Saver()

        for i in range(max_epoch):
            for j in range(train_interval):
                batch_data = dataset.next_batch_train()
                batch_code, summaries, this_loss, _ = self.sess.run([self.out_tensor, summary_op, self.loss, self.opt],
                                                                    feed_dict={self.img_in: batch_data})
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
            saver.save(self.sess, save_path + '\\hehemodel', global_step=step)
            print(str(mean_average_precision))
            dataset.reshuffle()
            del eval_to_write, mean_average_precision
            gc.collect()

    def _forward(self, batch_data):
        return self.sess.run(self.out_tensor, feed_dict={self.img_in: batch_data})
