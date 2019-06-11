import numpy as np
import tensorflow as tf
from PIL import ImageOps

from tf_mnist.model import IMAGE_SIZE, IMAGE_SIZE_SQUARED


def load_graph(filename):
    """Unpersists graph from file as default graph."""
    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    return graph_def


class Predictor:
    input_layer_name = 'input/x-input:0'
    output_layer_name = 'output:0'

    def __init__(self, model_filename='model.pb'):
        self.sess = tf.Session()
        self.labels = list(range(10))
        load_graph(model_filename)
        self.output_tensor = self.sess.graph.get_tensor_by_name(self.output_layer_name)
        self.dropout = self.sess.graph.get_tensor_by_name('dropout/Placeholder:0')

    def _process_image(self, img):
        """
        Preprocess an input image to a Numpy vector to be passed to Tensorflow.

        :param img: Input image
        :type img: PIL.Image.Image
        :return: Tuple of vector and a flag indicating whether we inverted the data during processing
        :rtype: Tuple[np.array, bool]
        """
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('L')
        if img.getpixel((0, 0)) > 192:
            # If the image looks like it's dark-on-white, invert it first
            img = ImageOps.invert(img)
            inverted = True
        else:
            inverted = False
        image_data = np.reshape(np.array(img, dtype='float32'), (1, IMAGE_SIZE_SQUARED))
        # Normalize the data to 0..1
        max_value = np.max(image_data)
        if max_value != 0:
            image_data /= max_value
        return image_data, inverted

    def predict_digit(self, img, digits=6):
        """
        Predict the digit from the given image using a pre-trained network.
        :param img: PIL image of a handwritten digit.
        :param digits: decimal points for prediction rounding
        :return: Dictionary describing the prediction
        :rtype: dict
        """
        image_data, inverted = self._process_image(img)
        predictions, = self.sess.run(self.output_tensor, {self.input_layer_name: image_data, self.dropout: 1})
        prediction_output = dict([
            (self.labels[top_pred_node_id], round(float(predictions[top_pred_node_id]), digits))
            for top_pred_node_id
            in predictions.argsort()[::-1]
        ])
        best_guess_value, best_guess_probability = max(prediction_output.items(), key=lambda item: item[1])
        return {
            'best_guess': best_guess_value,
            'best_guess_probability': best_guess_probability,
            'predictions': prediction_output,
            'inverted': inverted,
        }

    def close(self):
        if self.sess:
            self.sess.close()
