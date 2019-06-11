from __future__ import absolute_import, division, print_function

import json
import os
from shutil import copy2

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tf_mnist.utils import get_first_file


def read_inputs(flags):
    inputs_dir = os.getenv('VH_INPUTS_DIR', './inputs')
    data_set_files = {
        'train-images-idx3-ubyte.gz': get_first_file(os.path.join(inputs_dir, 'training-set-images')),
        'train-labels-idx1-ubyte.gz': get_first_file(os.path.join(inputs_dir, 'training-set-labels')),
        't10k-images-idx3-ubyte.gz': get_first_file(os.path.join(inputs_dir, 'test-set-images')),
        't10k-labels-idx1-ubyte.gz': get_first_file(os.path.join(inputs_dir, 'test-set-labels')),
    }
    train_dir = os.getcwd()
    for filename, src_path in data_set_files.items():
        dst_path = os.path.join(train_dir, filename)
        copy2(src_path, dst_path)

    return input_data.read_data_sets(train_dir, fake_data=flags.fake_data)


def save_output(sess, all_weights, all_biases):
    # Saving weights and biases as outputs of the task.
    outputs_dir = os.getenv('VH_OUTPUTS_DIR', './outputs')

    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)

    for i, ws in enumerate(all_weights):
        filename = os.path.join(outputs_dir, 'layer-{}-weights.csv'.format(i))
        np.savetxt(filename, ws.eval(), delimiter=",")
    for i, bs in enumerate(all_biases):
        filename = os.path.join(outputs_dir, 'layer-{}-biases.csv'.format(i))
        np.savetxt(filename, bs.eval(), delimiter=",")

    # Save the graph.
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph.as_graph_def(),
        output_node_names=['output'],  # TODO: is this even correct
    )
    with tf.gfile.FastGFile(os.path.join(outputs_dir, 'model.pb'), 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def train(flags, sess, model, mnist):
    x = model['x']
    y_ = model['y_']
    keep_prob = model['keep_prob']
    accuracy = model['accuracy']
    train_step = model['train_step']
    merged = model['merged']
    all_weights = model['all_weights']
    all_biases = model['all_weights']
    train_writer = tf.summary.FileWriter(flags.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(flags.log_dir + '/test')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or flags.fake_data:
            xs, ys = mnist.train.next_batch(batch_size=flags.batch_size, fake_data=flags.fake_data)
            k = flags.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(flags.max_steps):

        if i % 10 == 0:
            # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print(json.dumps({'step': i, 'accuracy': acc.item()}))
        elif i % 100 == 99:  # Record train set summaries, and train
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run(
                [merged, train_step],
                feed_dict=feed_dict(True),
                options=run_options,
                run_metadata=run_metadata,
            )
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        else:
            # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

    _, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    print(json.dumps({'step': flags.max_steps, 'accuracy': acc.item()}))

    train_writer.close()
    test_writer.close()

    save_output(sess, all_weights, all_biases)
