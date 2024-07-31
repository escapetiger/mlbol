import os
import importlib
import importlib.util
from types import ModuleType

__all__ = ["import_module", "import_all_modules"]


def import_module(path: str, name: str) -> ModuleType:
    file = name + ".py"
    path = os.path.join(path, file)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_all_modules(folder: str, name: str) -> None:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".py") and file[:-3] == name:
                path = os.path.join(root, file)
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
