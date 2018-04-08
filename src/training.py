from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import time
import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from model import Classifier


SIZE = 28
DATA_PATH = '../data/ant_img_gs'
SAVE_PATH = '../data/tf_save/trained_model/model.ckpt'


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


def main(args):
    print('Reading data...')
    data = get_data(args.data_path, args.train_set_size)

    classifier = Classifier(args.image_size, train=True)

    print('Training...')
    start_time = time.time()

    train_interval = 100
    for i in range(0, args.steps, train_interval):
        classifier.train(data['train']['images'], data['train']['labels'], steps=train_interval)
        classifier.save_checkpoint(args.save_path)

        delta_time = time.time() - start_time
        time_message = '{} s'.format(
            int(delta_time)) if delta_time > 1 else '{} ms'.format(int(delta_time * 1000))

        train_accuracy = classifier.evaluate(data['train']['images'], data['train']['labels'])
        test_accuracy = classifier.evaluate(data['test']['images'], data['test']['labels'])

        print('{}/{} Accuracy: Train: {:.4f}, Test {:.4f}. {}'.format(
            i + 1, args.steps, train_accuracy, test_accuracy, time_message))
        start_time = time.time()

    classifier.close()


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
                        default=1,
                        help='relative size of train set to whole data set')
    parser.add_argument('--steps', type=int,
                        default=3000,
                        help='number of training steps')

    args = parser.parse_args()

    main(args)
