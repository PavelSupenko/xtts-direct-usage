import os
from pathlib import Path


def prepare_directory(directory_path):
    if not os.path.exists(directory_path):
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
    else:
        clear_directory(directory_path)


def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
