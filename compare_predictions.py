import json
import os
import shutil
from collections import namedtuple

from numpy import float32

import valohai


def main():
    # valohai.prepare enables us to update the valohai.yaml configuration file with
    # the Valohai command-line client by running `valohai yaml step compare_predictions.py`

    valohai.prepare(
        step='compare-predictions',
        image='python:3.9',
        default_inputs={
            'predictions': {
                'default': None,
                'optional': False,
            },
            'models': [],
        },
    )

    # here we have some simple example logic to compare predictions to figure out which
    # predictions are the best, so this varies from use-case to use-case
    BestModel = namedtuple('BestModel', 'prediction, average_best_guess, model')
    best_of_best = BestModel(prediction=None, average_best_guess=None, model=None)
    average_best_guesses = dict()
    model_filename = ''

    for prediction_path in valohai.inputs('predictions').paths():
        filename = os.path.basename(prediction_path)

        extension = os.path.splitext(prediction_path)[1].lower()
        if extension != '.json':
            print(f'{filename} is not a JSON file')
            continue

        with open(prediction_path, 'r') as file:
            blob = json.load(file)

        best_guess_probabilities = []
        for sample_filename, prediction in blob.items():
            best_guess = str(prediction['best_guess'])
            probability = prediction['predictions'][best_guess]
            best_guess_probabilities.append(float32(probability))

        average_best_guess = sum(best_guess_probabilities) / len(best_guess_probabilities)
        average_best_guesses[filename] = average_best_guess
        print(f'{filename} => {average_best_guess} (average best guess probability)')

        suffix = filename.split('predictions-')[1].split('.json')[0]
        model_filename = f"model-{suffix}.h5"

        if not best_of_best.average_best_guess or average_best_guess > best_of_best.average_best_guess:
            best_of_best = BestModel(
                prediction=filename,
                average_best_guess=average_best_guess,
                model=model_filename,
            )

    print(f'The best model is the one that generated {best_of_best.prediction} ({best_of_best.average_best_guess})')

    model_path = next((model for model in valohai.inputs('models').paths() if model_filename in model), '')
    if model_path:
        shutil.copy(model_path, valohai.outputs().path(model_filename))


if __name__ == '__main__':
    main()
