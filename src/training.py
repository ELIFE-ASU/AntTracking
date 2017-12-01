import time
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf


SIZE = 28
DATA_PATH = '../data/ant_img_gs'
SAVE_PATH = '../data/tf_save/trained_model.ckpt'


def get_data(path, split):
    images = []
    labels = []

    type_labels = {'individual': [1, 0, 0],
                   'tandem': [0, 1, 0],
                   'transport': [0, 0, 1]}

    for img_type in type_labels:
        sub_path = os.path.join(path, img_type)
        for img_file in os.listdir(sub_path):
            if not img_file.startswith('.') and img_file.endswith('.png'):
                im = cv2.imread(os.path.join(sub_path, img_file), 0)
                if im.shape != (SIZE, SIZE):
                    continue
                # Normalize
                # mask = im != 0
                # im = mask * (im / 256.)

                images.append(im)
                labels.append(type_labels[img_type])

    images = np.asarray(images)
    labels = np.asarray(labels)

    perm = np.random.permutation(len(images))
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]

    training_size = int(len(images) * split)

    training_images = shuffled_images[:training_size]
    training_labels = shuffled_labels[:training_size]

    test_images = shuffled_images[training_size:]
    test_labels = shuffled_labels[training_size:]

    return {'train': {'images': training_images, 'labels': training_labels},
            'test': {'images': test_images, 'labels': test_labels}}


def build_cnn(size):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    x = tf.placeholder(tf.float32, shape=[None, size, size])
    y_ = tf.placeholder(tf.float32, [None, 3])

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, size, size, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 8])
        # b_conv = bias_variable([8])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(
            x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME'))

    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable[(5, 5, 1, 16)]
        h_conv2 = tf.nn.relu(tf.nn.conv2d(
            h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME'))

    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[
                                 1, 2, 2, 1], padding='SAME')

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([size // 2 * size // 2 * 16, 100])
        b_fc1 = bias_variable([100])

        h_pool_flat = tf.reshape(h_pool2, [-1, size // 2 * size // 2 * 8])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([100, 3])
        b_fc2 = bias_variable([3])

        y = tf.matmul(h_fc1, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    return x, y, y_, train_step


def main(args):
    print('Reading data...')
    data = get_data(args.data_path, args.train_set_size)

    x, y, y_, train_step = build_cnn(args.image_size)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Training...')
    start_time = time.time()
    steps = args.steps
    for i in range(steps):
        if (i + 1) % 100 == 0:
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            delta_time = time.time() - start_time
            time_message = '{} s'.format(
                int(delta_time)) if delta_time > 1 else '{} ms'.format(int(delta_time * 1000))

            print('{}/{} Training accuracy {}. {}'.format(
                i + 1,
                steps,
                sess.run(accuracy, feed_dict={
                    x: data['train']['images'], y_: data['train']['labels']}),
                time_message))
            start_time = time.time()

            saver.save(sess, args.save_path)

        sess.run(train_step, feed_dict={
            x: data['train']['images'], y_: data['train']['labels']})

    if args.train_set_size < 1:
        print('Test accuracy {}'.format(sess.run(accuracy, feed_dict={
            x: data['test']['images'], y_: data['test']['labels']})))
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default=DATA_PATH,
                        help='data path to training images')
    parser.add_argument('--image_size', type=int,
                        default=SIZE,
                        help='size of square image')
    parser.add_argument('--save_path', type=str,
                        default=SAVE_PATH,
                        help='TensorFlow checkpoint save path')
    parser.add_argument('--train_set_size', type=float,
                        default=0.9,
                        help='relative size of train set to whole data set')
    parser.add_argument('--steps', type=int,
                        default=3000,
                        help='number of training steps')

    args = parser.parse_args()

    main(args)
