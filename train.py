# Original:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
from shutil import copy2
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import get_first_file

FLAGS = None

def create_log_snapshot(file_writer):
    file_writer.close()
    # Set all files as read-only to inform Valohai they are done
    for subdir, dirs, files in os.walk(file_writer.get_logdir()):
        for f in files:
            os.chmod(subdir + os.sep + f, 292)
    file_writer.reopen()

def train():
    # Import input data
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '/tmp/tensorflow/mnist/inputs')
    data_set_files = [
        get_first_file(os.path.join(INPUTS_DIR, 'training-set-images')),
        get_first_file(os.path.join(INPUTS_DIR, 'training-set-labels')),
        get_first_file(os.path.join(INPUTS_DIR, 'test-set-images')),
        get_first_file(os.path.join(INPUTS_DIR, 'test-set-labels')),
    ]
    train_dir = os.getcwd()
    for file in data_set_files:
        copy2(file, train_dir)
    mnist = input_data.read_data_sets(train_dir, fake_data=FLAGS.fake_data)

    sess = tf.InteractiveSession()

    # Create a multilayer model.

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    all_weights = []
    all_biases = []

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
                all_weights.append(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
                all_biases.append(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train \
            .AdamOptimizer(FLAGS.learning_rate) \
            .minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps):

        if i % 10 == 0:
            # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print(json.dumps({'step': i, 'accuracy': acc.item()}))
        else:
            # Record train set summaries, and train
            if i % 100 == 99:
                # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)

                create_log_snapshot(train_writer)

                print('Adding run metadata for', i)
            else:
                # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)

    _, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    print(json.dumps({'step': FLAGS.max_steps, 'accuracy': acc.item()}))

    train_writer.close()
    test_writer.close()

    # Saving weights and biases as outputs of the task.
    outputs_dir = os.getenv('VH_OUTPUTS_DIR', '/tmp/tensorflow/mnist/outputs')
    for i, ws in enumerate(all_weights):
        filename = os.path.join(outputs_dir, 'layer-{}-weights.csv'.format(i))
        np.savetxt(filename, ws.eval(), delimiter=",")
    for i, bs in enumerate(all_biases):
        filename = os.path.join(outputs_dir, 'layer-{}-biases.csv'.format(i))
        np.savetxt(filename, bs.eval(), delimiter=",")


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Number of steps to run trainer')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout')
    parser.add_argument('--fake_data', type=bool, nargs='?', const=True, default=False,
                        help='If true, uses fake data for unit testing')
    parser.add_argument('--log_dir', type=str, default='/valohai/outputs/logs',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
