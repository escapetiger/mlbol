import os
import warnings
import pathlib
from mlbol.utils import Registry
from mlbol.utils import GroupedRegistry
from mlbol.utils import Dispatcher
from mlbol.utils import classproperty
from mlbol.utils import import_module


class _DTensorEngine(Dispatcher):
    """Dense tensor engine based on third-party packages.

    The main functionality are listed as follows:
    * Switch backend to enable tensor manipulation with the help of different packages.
    * Dynamicaly dispatch attributes/methods at runtime.
    """

    _environment = "MLBOL_BACKEND"
    _loaded_backends = Registry("backend")
    _available_backend_names = ["numpy", "pytorch"]
    _default_backend_name = "pytorch"

    @classproperty
    def available_backends(cls):
        """List available backend name."""
        return cls._available_backend_names

    @classmethod
    def get_backend_name(cls) -> str:
        """Return current backend name."""
        return cls._get_registry().name

    @classmethod
    def register_backend(cls, name: str, registry: GroupedRegistry) -> None:
        """Register backend to `_loaded_backends`."""
        cls._loaded_backends.register_(name, registry)

    @classmethod
    def initialize(cls) -> None:
        """Initialize the backend dispatcher.

        1) Retrieve the default registry name from the system environment variable
           If not found, use _default_backend_name instead.
        2) Set the registry by the retrieved registry name.
        """
        backend_name = os.environ.get(cls._environment, cls._default_backend_name)
        if backend_name not in cls._available_backend_names:
            msg = (
                f"{cls._environment} should be one of {''.join(map(repr, cls._available_backend_names))}"
                f", got {backend_name}. Defaulting to {cls._default_backend_name}'"
            )
            warnings.warn(msg, UserWarning)
            backend_name = cls._default_backend_name

        cls._default_backend_name = backend_name
        cls.set_backend(backend_name)

    @classmethod
    def load_backend(cls, backend_name: str) -> GroupedRegistry:
        """Load an existing backend or register a new backend
        by importing the corresponding module.

        Parameters
        ----------
        backend_name : str
            Name of the backend to load.

        Returns
        -------
        GroupedRegistry
            Backend registry.

        Raises
        ------
        ValueError
            If `backend_name` is not available.
        """
        api = pathlib.Path(__file__).parent.parent / "api" / "dtensor"

        if backend_name not in cls._available_backend_names:
            msg = f"Unknown backend name {backend_name!r}, known backends are {cls._available_backend_names}"
            raise ValueError(msg)
        if backend_name not in cls._loaded_backends:
            module = import_module(api, backend_name)
            backend = getattr(module, "backend")
            cls.register_backend(backend_name, backend)

        return cls._loaded_backends.get(backend_name)

    @classmethod
    def set_backend(cls, backend_name: str, threadsafe: bool = False) -> None:
        """Changes the registry to the specified one.

        Parameters
        ----------
        backend: str
            Name of the backend to load.
        threadsafe : bool, optional, default is False
            If False, set the backend as default for all threads.
        """
        if not isinstance(backend_name, str):
            raise TypeError(
                f"backend_name should be a string but not {type(backend_name).__name__}."
            )
        if backend_name not in cls._loaded_backends:
            backend_registry = cls.load_backend(backend_name)
        else:
            backend_registry = cls._loaded_backends.get(backend_name)
        super()._set_registry(backend_registry, threadsafe=threadsafe)

