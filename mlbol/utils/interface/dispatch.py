import threading
import inspect
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Optional
from mlbol.utils.interface.registry import GroupedRegistry

__all__ = ["Dispatcher"]


class _DynamicAttribute:
    """Dynamically dispatched attributes used by `Dispatcher`."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, cls: Optional[Any] = None) -> Any:
        if obj is None:
            # Accessing the attribute from the class level
            return cls._get_registry().get(self.name, group="A")
        else:
            # Accessing the attribute from an instance
            return obj._get_registry().get(self.name, group="A")


class Dispatcher(ModuleType):
    """Dispatcher base class, supporting statically or dynamically
    dispatch attributes and methods at runtime level.

    .. note::

        Do not use this class to create instances!
    """

    __slots__ = ("_registry", "_thread_local_data")
    _registry: GroupedRegistry
    _thread_local_data: threading.local

    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)
        cls._registry = GroupedRegistry(group=("A", "M"))
        cls._thread_local_data = threading.local()

    def __dir__(cls) -> None:
        attributes = cls._get_registry().keys(group="A")
        methods = cls._get_registry().keys(group="M")
        return list(cls.__dict__.keys()) + attributes + methods

    @property
    def registry(self) -> GroupedRegistry:
        return self._registry

    @registry.setter
    def registry(self, v: GroupedRegistry) -> None:
        self._registry = v

    @classmethod
    def _get_registry(cls) -> GroupedRegistry:
        """Returns the currently used registry."""
        return cls._thread_local_data.__dict__.get("registry", cls._registry)

    @classmethod
    def _set_registry(cls, registry: GroupedRegistry, threadsafe: bool = False) -> None:
        """Changes the registry to the specified one.

        Parameters
        ----------
        registry: GroupedRegistry
            GroupedRegistry instance.
        threadsafe : bool, optional, default is False
            If False, set the registry as default for all threads.
        """
        setattr(cls._thread_local_data, "registry", registry)
        if not threadsafe:
            cls._registry = registry

    @classmethod
    def _static_method(cls, method: Callable) -> staticmethod:
        return staticmethod(method)

    @classmethod
    def _dynamic_method(cls, name: str, method: Callable) -> staticmethod:
        """Create a dispatched function from a generic method."""

        def wrapper(*args, **kwargs):
            """A dynamically dispatched method.

            Returns the queried method from the currently registry.
            """
            return cls._get_registry().get(name, group="M")(*args, **kwargs)

        # We don't use `functools.wraps` here because some of the dispatched
        # methods include the backend (`cls`) as a parameter. Instead we manually
        # copy over the needed information, and filter the signature for `cls`.
        for attr in [
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
        ]:
            try:
                setattr(wrapper, attr, getattr(method, attr))
            except AttributeError:
                pass

        getattr(wrapper, "__dict__").update(getattr(method, "__dict__", {}))
        wrapper.__wrapped__ = method
        try:
            sig = inspect.signature(method)
            if "self" in sig.parameters:
                parameters = [v for k, v in sig.parameters.items() if k != "self"]
                sig = sig.replace(parameters=parameters)
            wrapper.__signature__ = sig
        except ValueError:
            # If it doesn't have a signature we don't need to remove self
            # This happens for NumPy (e.g. np.where) where inspect.signature(np.where) errors:
            # ValueError: no signature found for builtin <built-in function where>
            pass

        return staticmethod(wrapper)

    @classmethod
    def _static_dispatch(cls, group: str, mod: ModuleType | None = None) -> None:
        if not mod:
            mod = cls
        for k, v in cls._get_registry().grouped_by(group=group):
            if group in ["A"]:
                setattr(mod, k, v)
            if group in ["M"]:
                setattr(mod, k, cls._static_method(v))

    @classmethod
    def _dynamic_dispatch(cls, group: str, mod: ModuleType | None = None) -> None:
        if mod:
            for k, v in cls._get_registry().grouped_by(group=group):
                if group in ["A"]:
                    raise ValueError(
                        "Can not dynamic dispatch attributes to an external module!"
                    )
                if group in ["M"]:
                    setattr(mod, k, cls._dynamic_method(k, v))
        else:
            for k, v in cls._get_registry().grouped_by(group=group):
                if group in ["A"]:
                    setattr(cls, k, _DynamicAttribute(k))
                if group in ["M"]:
                    setattr(cls, k, cls._dynamic_method(k, v))

    @classmethod
    def dispatch(
        cls,
        group: str,
        target: ModuleType | None = None,
        is_static: bool = False,
    ) -> None:
        """Dispatch attributes or methods to target `module`.

        Parameters
        ----------
        group : {'A', 'M'}
            Group index. 'A' is for attributes and 'M' is for methods.
        target : ModuleType | None, optional
            Target module. If None (default), target is set to `cls`.
        is_static : bool, optional
            If False (default), static dispatch, and vice versa.
        """
        if is_static:
            cls._static_dispatch(group, mod=target)
        else:
            cls._dynamic_dispatch(group, mod=target)
