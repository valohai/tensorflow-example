import json

from PIL import Image
from werkzeug.debug import DebuggedApplication
from werkzeug.wrappers import Request, Response

from tf_mnist.predict import Predictor

predictor = None


def read_image_from_wsgi_request(environ):
    request = Request(environ)
    if not request.files:
        return None
    file_key = list(request.files.keys())[0]
    file = request.files.get(file_key)
    img = Image.open(file.stream)
    img.load()
    return img


def predict_wsgi(environ, start_response):
    img = read_image_from_wsgi_request(environ)
    if not img:
        return Response('no file uploaded', 400)(environ, start_response)

    # The predictor must be lazily instantiated;
    # the TensorFlow graph can apparently not be shared
    # between processes.
    global predictor
    if not predictor:
        predictor = Predictor()
    prediction = predictor.predict_digit(img)
    response = Response(json.dumps(prediction), mimetype='application/json')
    return response(environ, start_response)


predict_wsgi = DebuggedApplication(predict_wsgi)

if __name__ == '__main__':
    from werkzeug.serving import run_simple

    run_simple('localhost', 3000, predict_wsgi)
