import os


def get_first_file(path):
    filename = os.listdir(path)[0]
    return os.path.join(path, filename)
