import argparse
import glob
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prediction-dir', required=True)
    args = ap.parse_args()
    vh_inputs_dir = os.getenv('VH_INPUTS_DIR', './')
    vh_outputs_dir = os.getenv('VH_OUTPUTS_DIR', './')

    if not os.path.isdir(args.prediction_dir):
        raise Exception('--prediction-dir must be a directory')

    json_files = glob.glob('{}/*.json'.format(args.prediction_dir))
    if not json_files:
        raise Exception('no .json predictions under {}'.format(json_files))

    print('Comparing predictions of {} JSON files...'.format(len(json_files)))

    prediction_blobs = dict()
    for file_path in json_files:
        with open(file_path, 'r') as f:
            prediction_blobs[os.path.basename(file_path)] = json.load(f)

    # here we have some simple example logic to compare predictions to figure out which
    # predictions are the best, so this varies from use-case to use-case
    best_of_best = (None, None, None)
    average_best_guesses = dict()
    for prediction_filename, blob in prediction_blobs.items():
        best_guess_probabilities = []
        for sample_filename, prediction in blob.items():
            best_guess = str(prediction['best_guess'])
            probability = prediction['predictions'][best_guess]
            best_guess_probabilities.append(probability)
        average_best_guess = sum(best_guess_probabilities) / len(best_guess_probabilities)
        average_best_guesses[prediction_filename] = average_best_guess
        print('{} => {} (average best guess probability)'.format(prediction_filename, average_best_guess))

        suffix = prediction_filename.split('predictions-')[1].split('.json')[0]
        model_filename = ("model-{}.pb").format(suffix)
        model_filepath = os.path.join(vh_inputs_dir, 'models', model_filename)

        if not best_of_best[1]:
            best_of_best = (prediction_filename, average_best_guess, model_filename)
        elif average_best_guess > best_of_best[1]:
            best_of_best = (prediction_filename, average_best_guess, model_filename)

    print('The best model is the one that generated {} ({})'.format(best_of_best[0], best_of_best[1]))
    
    if(os.path.exists(model_filepath)) :
        shutil.copy(model_filepath, os.path.join(vh_outputs_dir, 'model.pb'))


if __name__ == '__main__':
    main()
