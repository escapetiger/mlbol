import sys

from mlbol.vision._engine import _VisionEngine as _vision_engine

from mlbol.vision._attributes import plt

from mlbol.vision._methods import multi_plot
from mlbol.vision._methods import multi_plot_by_files
from mlbol.vision._methods import multi_imshow
from mlbol.vision._methods import multi_imshow_by_files


_vision_engine.initialize()
_vision_engine.dispatch("A", target=sys.modules[__name__], is_static=True)
_vision_engine.dispatch("M", target=None, is_static=False)


def set_vision_backend(backend_name: str, threadsafe: bool = False) -> None:
    """Set backend for vision.

    One must manually dispatch attributes in static mode to this module.

    Parameters
    ----------
    backend: {'numpy', 'pytorch'}
        Name of the backend to load.
    threadsafe : bool, optional, default is False
        If False, set the backend as default for all threads.
    """
    _vision_engine.set_backend(backend_name, threadsafe=threadsafe)
    _vision_engine.dispatch("A", target=sys.modules[__name__], is_static=True)
