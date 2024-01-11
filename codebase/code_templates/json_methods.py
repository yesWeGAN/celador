# collection of methods revolving around json files
import json
import os
from pathlib import Path
from datetime import datetime


def load_json_file(path, filename):
    """
    Loads all content from a given json-filename in a directory.
    :param path: Path | str
    :param filename: str
    :return: dict
    """
    try:
        filep = next(Path(path).rglob(f"*{filename}"))
    except StopIteration:
        raise FileNotFoundError(f"No json file found for {filename}. Exiting.")

    return json.load(open(filep, "r"))


def save_to_json_file(dictionary, path, filename=None, makedirs=False):
    """
    Save dictionary to a given json-filename in a directory.
    :param makedirs: bool
    :param dictionary: dict
    :param path: Path | str
    :param filename: str
    """
    if not makedirs:
        if not os.path.isdir(path):
            raise NotADirectoryError(
                f"Not a directory: {path}. Pass a valid path or enable makedirs=True."
            )
    else:
        os.makedirs(path, exist_ok=True)

    if filename is None:
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        filename = f"json_file_{date_time}"

    json.dump(dictionary, open(os.path.join(path, filename), "w"))


class CustomClass:
    def __init__(self):
        self.attr = None
        self.input = Path(".")


def serialize(objectlike):
    """
    Prepare any object and all its inner vars for serialization in json.
    :param objectlike: object
    :return: dict
    """
    output = {}
    for attr, state in vars(objectlike).items():
        if isinstance(state, dict):
            keyset = [
                key.as_posix() if isinstance(key, Path) else key for key in state.keys()
            ]
            valueset = [
                value.as_posix() if isinstance(value, Path) else value
                for value in state.values()
            ]
            output[attr] = dict(zip(keyset, valueset))

        elif isinstance(state, Path):
            output[attr] = str(state.as_posix())

        elif isinstance(state, CustomClass):
            continue
        else:
            output[attr] = state

    return dict(sorted(output.items()))
