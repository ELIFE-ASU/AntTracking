from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import tensorflow as tf


class Classifier:
    def __init__(self, size, train=False):
        graph = Classifier.build_graph(size, train)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.x = graph['x']
        self.y = graph['y']

        if train:
            self.y_ = graph['y_']
            self.accuracy = graph['accuracy']
            self.train_step = graph['train_step']

            self.sess.run(tf.global_variables_initializer())

    def save_checkpoint(self, path):
        self.saver.save(self.sess, path)

    def load_checkpoint(self, path):
        self.saver.restore(self.sess, path)

    def train(self, data, labels, steps):
        try:
            for _ in range(steps):
                self.sess.run(self.train_step, feed_dict={self.x: data, self.y_: labels})
        except AttributeError:
            raise AttributeError('Classifier not set to be trainable')

    def evaluate(self, data, labels):
        try:
            return self.sess.run(self.accuracy, feed_dict={self.x: data, self.y_: labels})
        except AttributeError:
            raise AttributeError('Classifier not set to be trainable')

    def classify(self, data):
        return self.sess.run(self.y, feed_dict={self.x: data})

    def close(self):
        self.sess.close()

    @staticmethod
    def build_graph(size, train=False):
        """Build a TensorFlow graph as the classifier."""
        # CNN
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        tf.reset_default_graph()

        graph = {}

        x = tf.placeholder(tf.float32, shape=[None, size, size])

        with tf.name_scope('reshape'):
            x_image = tf.reshape(x, [-1, size, size, 1])

        with tf.name_scope('conv'):
            W_conv = weight_variable([3, 3, 1, 8])
            h_conv = tf.nn.relu(tf.nn.conv2d(
                x_image, W_conv, strides=[1, 1, 1, 1], padding='SAME'))

        with tf.name_scope('pool'):
            h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([size // 2 * size // 2 * 8, 100])
            b_fc1 = bias_variable([100])

            h_pool_flat = tf.reshape(h_pool, [-1, size // 2 * size // 2 * 8])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([100, 3])
            b_fc2 = bias_variable([3])

            y = tf.matmul(h_fc1, W_fc2) + b_fc2

        graph['x'] = x
        graph['y'] = y

        if train:
            y_ = tf.placeholder(tf.float32, [None, 3])

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            graph['y_'] = y_
            graph['accuracy'] = accuracy
            graph['train_step'] = train_step

        return graph
