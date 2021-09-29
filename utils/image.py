import numpy as np
from PIL import ImageOps

IMAGE_SIZE = 28
LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def predict_image(model, image, inverted=False):
    prediction = model.predict(image)[0]

    best_guess = np.argmax(prediction)
    best_guess_probability = str(prediction[best_guess])
    predictions = {str(label): str(prediction[label]) for label in LABELS}

    return {
        'best_guess': str(best_guess),
        'best_guess_probability': best_guess_probability,
        'predictions': predictions,
        'inverted': inverted,
    }


def process_image(image):
    # Resize the image to MNIST size 28 x 28 pixels
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Make the image grayscale
    image = image.convert('L')

    # If the image looks like it's dark-on-white, invert it first
    inverted = image.getpixel((0, 0)) > 192
    if inverted:
        print('Inverting image')
        image = ImageOps.invert(image)

    # Transform the image into a NumPy array
    image_data = np.array(image)

    # Normalize the pixel values from 0-255 to 0.0-1.0
    image_data = image_data / 255.0

    return image_data.reshape((1, IMAGE_SIZE, IMAGE_SIZE)), inverted
