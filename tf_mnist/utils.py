import os


def get_first_file(path, raise_on_missing=True):
    files = os.listdir(path)
    if not files and raise_on_missing:
        raise Exception('no files in {}'.format(path))
    filename = os.listdir(path)[0]
    return os.path.join(path, filename)
