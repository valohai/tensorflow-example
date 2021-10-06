import tensorflow as tf


def load_model(path):
    return tf.keras.Sequential([
        tf.keras.models.load_model(path),
        tf.keras.layers.Softmax(),
    ])
