import argparse
import csv
import json
import os
import sys
import glob
import uuid

from PIL import Image

from tf_mnist.predict import Predictor

IMAGE_EXTENSIONS = {
    '.png',
    '.jpg',
    '.jpeg',
    '.bmp',
    '.gif',
    '.tiff',
}


def find_images(base):
    for dir_path, dir_names, filenames in os.walk(base):
        dir_names[:] = [dirname for dirname in dir_names if dir_names[0] not in '._']
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            base_file_path = os.path.relpath(file_path, start=base)
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            yield (base_file_path, file_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--output-dir', default=os.environ.get('VH_OUTPUTS_DIR', '.'))
    args = ap.parse_args()

    # validate the arguments
    if not os.path.isdir(args.model_dir):
        raise Exception('--model-dir must be a directory')
    if not os.path.isdir(args.image_dir):
        raise Exception('--image-dir must be a directory')
    if not os.path.isdir(args.output_dir):
        raise Exception('--output-dir must be a directory')

    # use the first model we find under the designated model directory
    model_files = glob.glob('{}/*.pb'.format(args.model_dir))
    if not model_files:
        raise Exception('no .pb models under {}'.format(model_files))
    model_filename = model_files[0]
    print('Using {}...'.format(model_filename))

    # generate names for prediction files (CSV and JSON)
    suffix = uuid.uuid4()
    output_json_filename = os.path.join(args.output_dir, 'predictions-{}.json'.format(suffix))

    # do batch inference on the given images
    json_blob = {}
    predictor = Predictor(model_filename=model_filename)
    for image_filename, image_path in find_images(args.image_dir):
        try:
            img = Image.open(image_path)
            prediction = predictor.predict_digit(img)
            json_blob[image_filename] = prediction
            print(image_filename, prediction)
        except Exception as exc:
            json_blob[image_filename] = {'error': exc}
            print('Unable to process %s: %s' % (image_filename, exc), file=sys.stderr)
    predictor.close()

    # save predictions in a JSON file
    with open(output_json_filename, 'w', newline='') as json_out:
        json.dump(json_blob, json_out, sort_keys=True)


if __name__ == '__main__':
    main()
