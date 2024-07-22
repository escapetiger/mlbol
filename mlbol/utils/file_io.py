import logging
import yaml
import pathlib
from typing import Any
from typing import Dict

__all__ = ["mkdir", "load_yaml"]


def mkdir(path: str):
    """Recursively make a new directory at a path."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _eval_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in obj.items():
        if not isinstance(v, Dict):
            try:
                obj[k] = eval(v)
            except:
                pass
        else:
            obj[k] = _eval_dict(v)
    return obj


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as file:
            config = yaml.full_load(file)
        return _eval_dict(config)
    except FileNotFoundError as err:
        logging.error(f"File {path} does not exist.")
        raise
    except Exception as exc:
        logging.error(f"Exception happens when loading file {path}.")
        raise
