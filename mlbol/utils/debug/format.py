import re
from typing import Iterable
from typing import Union
from typing import Tuple
from typing import Any
from typing import overload

__all__ = ["add_prefix", "add_suffix", "snake_case", "camel_to_snake", "snake_to_camel"]


def add_prefix(name: str, prefix: Union[str, Tuple[str, ...]]) -> str:
    if not name.startswith(prefix):
        name = (prefix[0] if isinstance(prefix, Tuple) else prefix) + name
    return name


def add_suffix(name: str, suffix: Union[str, Tuple[str, ...]]) -> str:
    if not name.endswith(suffix):
        name = name + (suffix[0] if isinstance(suffix, Tuple) else suffix)
    return name


@overload
def snake_case(s: str) -> str: ...


@overload
def snake_case(s: Iterable[str]) -> str: ...


@overload
def snake_case(s: Union[str, Iterable[str]]) -> str:
    """Create a string in snake case.

    Parameters
    ----------
    s : str or Iterable[str]
    """
    if isinstance(s, str):
        return s.lower()
    elif isinstance(s, Iterable):
        return ("_".join((_ for _ in s))).lower()


def camel_to_snake(s: str) -> str:
    """Convert a string from cmael case to snake case.

    References
    ----------
    * https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def snake_to_camel(s: str, lower: bool = False) -> str:
    """Convert a string from snake case to (lower) camel case.

    References
    ----------
    * https://stackoverflow.com/questions/19053707/converting-snake-case-to-lower-camel-case-lowercamelcase
    """
    if lower:
        upper_camel = "".join(x.capitalize() for x in s.lower().split("_"))
        return s[0].lower() + upper_camel[1:]

    return "".join(_.capitalize() for _ in s.lower().split("_"))


def make_tuple(*a: Any) -> Tuple[Any, ...]:
    if not hasattr(a, "__len__"):
        if not isinstance(a, tuple):
            return (a,)
    b = []
    for x in a:
        if not isinstance(x, tuple | list):
            x = (x,)
        b.append(x)
    return b
