# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py

import itertools
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Tuple
from tabulate import tabulate

__all__ = ["Registry", "GroupedRegistry"]


class Registry(Iterable[Tuple[str, Any]]):
    """The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backend registry):

    .. code-block:: python

        BACKEND_REGISTRY = Registry('backend')

    To register an object:

    .. code-block:: python

        @BACKEND_REGISTRY.register()
        class Mybackend():
            ...

    Or:

    .. code-block:: python

        BACKEND_REGISTRY.register(Mybackend)
    """

    def __init__(self, name: str = "") -> None:
        """Initialize an instance.

        Parameters
        ----------
        name : str
            The name of this registry.
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def register_(self, name: str, obj: Any) -> None:
        """Register the given object under the given name."""
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        """Register the given object under the name `obj.__name__`.
        Can be used as either a decorator or not.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self.register_(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self.register_(name, obj)

    def get(self, name: str) -> Any:
        return self.__getitem__(name)

    def keys(self) -> list[str]:
        return list(self._obj_map.keys())

    def values(self) -> list[Any]:
        return list(self._obj_map.values())

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    def __getitem__(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    __str__ = __repr__


class GroupedRegistry(Iterable[Tuple[str, Iterable[Tuple[str, Any]]]]):
    """The registry that provides (group, name -> object) mapping, to support third-party
    users' custom modules.
    """

    def __init__(self, name: str = "", group: Tuple[str] = None) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._group: Tuple[str] = tuple(group)
        self._obj_map: Dict[str, Dict[str, Any]] = {g: {} for g in self._group}

    @property
    def name(self) -> str:
        return self._name

    @property
    def group(self) -> tuple[str]:
        return self._group

    def register_(self, group: str, name: str, obj: Any) -> None:
        """Register the given object under the given name within a group."""
        assert (
            group in self._group
        ), f"A group named '{group}' was not supported in `{self._name}` registry."
        assert (
            name not in self._obj_map
        ), f"An object named '{name}' was already registered in '{self._name}' registry."
        self._obj_map[group][name] = obj

    def register(self, group: str, obj: Any = None) -> Any:
        """Register the given object under the name `obj.__name__` within a group.
        Can be used as either a decorator or not.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self.register_(group, name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self.register_(group, name, obj)

    def get(self, name: str, group: str = None) -> Any:
        """Returns a value within registry."""
        return self.__getitem__(name, group)

    def keys(self, group: str = None) -> list[str]:
        """Returns keys in registry, optionally within a group."""
        if group:
            return list(self._obj_map[group].keys())
        return list(itertools.chain(list(self._obj_map[g].keys()) for g in self._group))

    def values(self, group: str = None) -> list[Any]:
        """Returns values in registry, optionally within a group."""
        if group:
            return list(self._obj_map[group].values())
        return list(
            itertools.chain(list(self._obj_map[g].values()) for g in self._group)
        )

    def grouped_by(self, group: str) -> Iterator[Tuple[str, Any]]:
        """Returns an iterator of dictionary within a group."""
        return self._obj_map[group].items()

    def __contains__(self, name: str) -> bool:
        for g in self._group:
            if name in self._obj_map[g]:
                return True
        return False

    def __repr__(self) -> str:
        headers = ["Groups", "Names", "Objects"]
        values = list(iter(self))
        table = tabulate(values, headers=headers, tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter((g, k, v) for g, w in self._obj_map.items() for k, v in w.items())

    def __getitem__(self, name: str, group: str = None) -> Any:
        ret = None
        if group:
            ret = self._obj_map[group].get(name)
        else:
            for g in self._group:
                ret = self._obj_map[g].get(name)
                if ret:
                    break
        if ret is None:
            if group:
                msg = f"No object named '{name}' found in '{group}' group of '{self._name}' registry!"
            else:
                msg = f"No object named '{name}' found in '{self._name}' registry!"
            raise KeyError(msg)
        return ret

    __str__ = __repr__
