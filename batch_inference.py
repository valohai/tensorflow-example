import json
import os
import sys

import valohai
from PIL import Image

from utils.image import predict_image, process_image
from utils.model import load_model


def main():
    # valohai.prepare enables us to update the valohai.yaml configuration file with
    # the Valohai command-line client by running `valohai yaml step batch_inference.py`

    valohai.prepare(
        step='batch-inference',
        image='tensorflow/tensorflow:2.6.0',
        default_inputs={
            'model': {
                'default': None,
                'optional': False,
            },
            'images': [
                'https://valohaidemo.blob.core.windows.net/mnist/four-inverted.png',
                'https://valohaidemo.blob.core.windows.net/mnist/five-inverted.png',
                'https://valohaidemo.blob.core.windows.net/mnist/five-normal.jpg',
            ],
        },
    )

    print('Loading model')
    model_path = valohai.inputs('model').path()
    model = load_model(model_path)

    json_blob = {}
    for image_path in valohai.inputs('images').paths():
        filename = os.path.basename(image_path)

        extension = os.path.splitext(image_path)[1].lower()
        if extension not in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
            print(f'{filename} is not an image file')
            continue

        print(f'Running inference for {filename}')
        try:
            image, inverted = process_image(Image.open(image_path))
            prediction = predict_image(model, image, inverted)
            json_blob[filename] = prediction
            print(filename, prediction)
        except Exception as exc:
            json_blob[filename] = {'error': exc}
            print(f'Unable to process {filename}: {exc}', file=sys.stderr)

    print('Saving predictions')
    suffix = ''
    try:
        suffix = f'-{model_path.split("model-")[1].split(".h5")[0]}'
    except IndexError:
        print(f'Unable to get suffix from {model_path}')

    json_path = os.path.join(valohai.outputs().path(f'predictions{suffix}.json'))
    with open(json_path, 'w') as json_file:
        json.dump(json_blob, json_file, sort_keys=True)


if __name__ == '__main__':
    main()
