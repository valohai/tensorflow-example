import argparse

from PIL import Image

from tf_mnist.predict import Predictor

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    args = ap.parse_args()
    predictor = Predictor()
    for filename in args.files:
        img = Image.open(filename)
        print(filename, predictor.predict_digit(img))
    predictor.close()
