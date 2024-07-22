# ==== ATTRIBUTES ====
__version__ = "0.0.1"

from . import utils
from . import dtensor
from . import geometry
from . import data
from . import vision


def set_backend(
    bkd_dtensor: str, bkd_vision: str | None = None, threadsafe: bool = False
) -> None:
    dtensor.set_dtensor_backend(bkd_dtensor, threadsafe=threadsafe)
    if bkd_vision is not None:
        vision.set_vision_backend(bkd_vision, threadsafe=threadsafe)


from mlbol.dtensor import get_dtensor_backend
from mlbol.dtensor import set_dtensor_backend
from mlbol.dtensor import dtensor_profile


def __getattr__(name):
    for package in [dtensor]:
        if hasattr(package, name):
            return getattr(package, name)
    raise AttributeError(f"module 'mlbol' has no attribute '{name}'")

