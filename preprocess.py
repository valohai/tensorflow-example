# Based on:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import gzip
import os

import numpy as np

from tf_mnist.utils import get_first_file


def _load_images(file):
    """
    :param file: A file object that can be passed into a gzip reader.
    :return: A 4D uint8 numpy array [index -> y -> x -> brightness]
    :raises ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting {}'.format(file.name))
    with gzip.GzipFile(fileobj=file) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, file.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def _load_labels(file):
    """
    :param file: A file object that can be passed into a gzip reader.
    :return: A 1D uint8 numpy array [index -> label between 0-9]
    :raises ValueError: If the bytestream doesn't start with 2049.
    """
    print('Extracting {}'.format(file.name))
    with gzip.GzipFile(fileobj=file) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, file.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def _save_images(output_file_path, image_data):
    """
    :param output_file_path: Where to save the archive, filename included.
    :param image_data: the 4D np array to save
    :return:
    """
    header = np.array([0x0803, len(image_data), 28, 28], dtype='>i4')
    with gzip.open(output_file_path, "wb") as f:
        f.write(header.tobytes())
        f.write(image_data.tobytes())


def _save_labels(output_file_path, label_data):
    """
    :param output_file_path: Where to save the archive, filename included.
    :param label_data: the numpy array to save.
    :return:
    """
    header = np.array([0x0801, len(label_data)], dtype='>i4')
    with gzip.open(output_file_path, "wb") as f:
        f.write(header.tobytes())
        f.write(label_data.tobytes())


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def main():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
    TRAIN_IMAGES_DIR = os.path.join(INPUTS_DIR, 'training-set-images')
    TRAIN_LABELS_DIR = os.path.join(INPUTS_DIR, 'training-set-labels')
    TEST_IMAGES_DIR = os.path.join(INPUTS_DIR, 'test-set-images')
    TEST_LABELS_DIR = os.path.join(INPUTS_DIR, 'test-set-labels')

    train_images_path = get_first_file(TRAIN_IMAGES_DIR)
    train_labels_path = get_first_file(TRAIN_LABELS_DIR)
    test_images_path = get_first_file(TEST_IMAGES_DIR)
    test_labels_path = get_first_file(TEST_LABELS_DIR)

    with open(train_images_path, 'rb') as f:
        train_images = _load_images(f)
    with open(train_labels_path, 'rb') as f:
        train_labels = _load_labels(f)
    with open(test_images_path, 'rb') as f:
        test_images = _load_images(f)
    with open(test_labels_path, 'rb') as f:
        test_labels = _load_labels(f)

    # Note that this is only for demoing purposes, only extracts and compresses the data.
    print('Applying feature extraction...')
    # TODO: For the sake of example, use actual images to generate features?
    # TODO: Add parameters to control the feature extraction?

    OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
    _save_images(os.path.join(OUTPUTS_DIR, 'mnist-train-images.gz'), train_images)
    print('25%')
    _save_labels(os.path.join(OUTPUTS_DIR, 'mnist-train-labels.gz'), train_labels)
    print('50%')
    _save_images(os.path.join(OUTPUTS_DIR, 'mnist-test-images.gz'), test_images)
    print('75%')
    _save_labels(os.path.join(OUTPUTS_DIR, 'mnist-test-labels.gz'), test_labels)
    print('100%')


if __name__ == '__main__':
    main()
