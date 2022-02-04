import shutil


def delete_temp_dir(directory):
    try:
        shutil.rmtree(directory)
    except Exception:
        pass


def unique(x):
    x = list(set(x))
    return x