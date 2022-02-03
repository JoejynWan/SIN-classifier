import shutil


def delete_temp_dir(directory):
    try:
        shutil.rmtree(directory)
    except Exception:
        pass