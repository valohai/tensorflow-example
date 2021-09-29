"""Runs prediction on images given as arguments.

Requires to have the trained MNIST model as a model.h5 file.
"""

import argparse

import tensorflow as tf
from PIL import Image

from utils.image import predict_image, process_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    args = ap.parse_args()

    model = tf.keras.models.load_model('model.h5')

    for filename in args.files:
        image, inverted = process_image(Image.open(filename))
        print(filename, predict_image(model, image, inverted))


if __name__ == '__main__':
    main()
