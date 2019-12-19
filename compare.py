import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prediction-dir', required=True)
    args = ap.parse_args()

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
    best_of_best = (None, None)
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

        if not best_of_best[1]:
            best_of_best = (prediction_filename, average_best_guess)
        elif average_best_guess > best_of_best[1]:
            best_of_best = (prediction_filename, average_best_guess)

    print('The best model is the one that generated {} ({})'.format(best_of_best[0], best_of_best[1]))


if __name__ == '__main__':
    main()
