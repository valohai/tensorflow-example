import argparse
import csv
import json
import os
import sys
import glob

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
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [dirname for dirname in dirnames if dirnames[0] not in ('._')]
        for filename in filenames:
            filename = os.path.join(dirpath, filename)
            relname = os.path.relpath(filename, start=base)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            yield (relname, filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--output-csv', default=os.path.join(os.environ.get('VH_OUTPUTS_DIR', '.'), 'predictions.csv'))
    ap.add_argument('--output-json', default=os.path.join(os.environ.get('VH_OUTPUTS_DIR', '.'), 'predictions.json'))
    args = ap.parse_args()

    # validate the arguments
    if not os.path.isdir(args.model_dir):
        raise Exception('--model-dir must be a directory')
    if not os.path.isdir(args.image_dir):
        raise Exception('--image-dir must be a directory')

    model_files = glob.glob(f'{args.model_dir}/*.pb')
    if not model_files:
        raise Exception(f'no .pb models under {model_files}')

    # use the first model we find...
    model_filename = model_files[0]
    print(f'Using {model_filename}...')

    predictor = Predictor(model_filename=model_filename)
    json_blob = {}
    with open(args.output_csv, 'w', newline='') as csv_out:
        csv_writer = csv.writer(csv_out)
        csv_writer.writerow([
            'filename',
            'best_guess',
            'best_guess_probability',
            'inverted',
            'error',
        ])

        for relname, filename in find_images(args.image_dir):
            try:
                img = Image.open(filename)
                prediction = predictor.predict_digit(img)
                print(relname, prediction)
                json_blob[relname] = prediction
                csv_writer.writerow([
                    relname,
                    prediction['best_guess'],
                    prediction['best_guess_probability'],
                    ('inverted' if prediction['inverted'] else ''),
                ])
            except Exception as exc:
                csv_writer.writerow([
                    relname,
                    '',
                    '',
                    '',
                    str(exc),
                ])
                json_blob[relname] = {'error': exc}
                print('Unable to process %s: %s' % (relname, exc), file=sys.stderr)

    predictor.close()

    with open(args.output_json, 'w', newline='') as json_out:
        json.dump(json_blob, json_out, sort_keys=True)


if __name__ == '__main__':
    main()
