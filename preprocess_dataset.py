import numpy as np
import os

def main():
    # Read input files from Valohai inputs directory
    # or local dir, if we're not in Valohai
    # /valohai/inputs/
    inputs_dir = os.getenv('VH_INPUTS_DIR', '.inputs')
    path_to_file = os.path.join(inputs_dir, 'dataset/mnist.npz')
    print("change!")
    print('Loading data')
    #with np.load(valohai.inputs('dataset').path(), allow_pickle=True) as file:
    with np.load(path_to_file, allow_pickle=True) as file:
        x_train, y_train = file['x_train'], file['y_train']
        x_test, y_test = file['x_test'], file['y_test']

    print('Preprocessing data')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Write output files to Valohai outputs directory
    # This enables Valohai to version your data
    # and upload output it to the default data store

    print('Saving preprocessed data')
    
    # Save generated artefacts to Valohai outputs
    # so they get versioned
    # or then use local directory
    # /valohai/outputs/
    output_dir = os.getenv('VH_OUTPUTS_DIR', '.outputs')
    path_to_output_file = os.path.join(output_dir, 'mnist.npz')

    #path_to_output_file = valohai.outputs().path('preprocessed_mnist.npz')
    np.savez_compressed(path_to_output_file, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    main()
