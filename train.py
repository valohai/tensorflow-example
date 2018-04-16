# Original:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

from __future__ import absolute_import, division, print_function

import argparse

import tensorflow as tf

from tf_mnist.model import make_model
from tf_mnist.training import read_input, train


def main(flags):
    sess = tf.InteractiveSession()
    mnist = read_input(flags)
    model = make_model(learning_rate=flags.learning_rate)
    train(flags, sess, model, mnist)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_steps',
        type=int,
        default=300,
        help='Number of steps to run trainer',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.9,
        help='Keep probability for training dropout',
    )
    parser.add_argument(
        '--fake_data',
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help='If true, uses fake data for unit testing',
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
        help='Summaries log directory',
    )
    flags = parser.parse_args()
    return flags


if __name__ == '__main__':
    flags = parse_args()
    main(flags)
