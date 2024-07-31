import pathlib
from mlbol.utils import Registry
from mlbol.utils import GroupedRegistry
from mlbol.utils import Dispatcher
from mlbol.utils import classproperty
from mlbol.utils import import_module


class _VisionEngine(Dispatcher):
    """Basic vision interface to third-party packages."""

    _loaded_backends = Registry("backend")
    _available_backend_names = ["matplotlib"]
    _default_backend_name = "matplotlib"

    @property
    def engine(self):
        return self

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
        """Initialize the backend dispatcher."""
        cls.set_backend(cls._default_backend_name)

    @classmethod
    def load_backend(cls, backend_name: str) -> GroupedRegistry:
        """Load backend with the given `backend_name`."""
        api = pathlib.Path(__file__).parent.parent / "api" / "vision"
        if backend_name not in cls._loaded_backends:
            module = import_module(api, backend_name)
            backend = getattr(module, "backend")
            cls.register_backend(backend_name, backend)
        return cls._loaded_backends.get(backend_name)

    @classmethod
    def set_backend(cls, backend_name: str, threadsafe: bool = False) -> None:
        """Set backend to be consistent with `dtensor_engine`."""
        if backend_name not in cls._loaded_backends:
            backend_registry = cls.load_backend(backend_name)
        else:
            backend_registry = cls._loaded_backends.get(backend_name)
        super()._set_registry(backend_registry, threadsafe=threadsafe)
