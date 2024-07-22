import sys
import contextlib
from typing import Generator

from mlbol.dtensor._engine import _DTensorEngine as _dtensor_engine

from mlbol.dtensor._attributes import int32
from mlbol.dtensor._attributes import int64
from mlbol.dtensor._attributes import float32
from mlbol.dtensor._attributes import float64
from mlbol.dtensor._attributes import complex64
from mlbol.dtensor._attributes import complex128
from mlbol.dtensor._attributes import e
from mlbol.dtensor._attributes import pi
from mlbol.dtensor._attributes import nan
from mlbol.dtensor._attributes import inf
from mlbol.dtensor._attributes import Tensor

from mlbol.dtensor._methods import get_default_tensor_context
from mlbol.dtensor._methods import set_default_tensor_context
from mlbol.dtensor._methods import get_tensor_dtype
from mlbol.dtensor._methods import get_tensor_device
from mlbol.dtensor._methods import is_tensor
from mlbol.dtensor._methods import context
from mlbol.dtensor._methods import tensor
from mlbol.dtensor._methods import as_tensor
from mlbol.dtensor._methods import to_numpy
from mlbol.dtensor._methods import copy
from mlbol.dtensor._methods import empty
from mlbol.dtensor._methods import empty_like
from mlbol.dtensor._methods import zeros
from mlbol.dtensor._methods import zeros_like
from mlbol.dtensor._methods import ones
from mlbol.dtensor._methods import ones_like
from mlbol.dtensor._methods import full
from mlbol.dtensor._methods import full_like
from mlbol.dtensor._methods import eye
from mlbol.dtensor._methods import diag
from mlbol.dtensor._methods import diagonal
from mlbol.dtensor._methods import arange
from mlbol.dtensor._methods import linspace
from mlbol.dtensor._methods import meshgrid
from mlbol.dtensor._methods import rand
from mlbol.dtensor._methods import randn
from mlbol.dtensor._methods import gamma
from mlbol.dtensor._methods import choice
from mlbol.dtensor._methods import randint
from mlbol.dtensor._methods import permutation
from mlbol.dtensor._methods import uniform
from mlbol.dtensor._methods import normal
from mlbol.dtensor._methods import glorot_uniform
from mlbol.dtensor._methods import glorot_normal
from mlbol.dtensor._methods import exponential
from mlbol.dtensor._methods import size
from mlbol.dtensor._methods import shape
from mlbol.dtensor._methods import ndim
from mlbol.dtensor._methods import index_update
from mlbol.dtensor._methods import reshape
from mlbol.dtensor._methods import transpose
from mlbol.dtensor._methods import ravel
from mlbol.dtensor._methods import ravel_multi_index
from mlbol.dtensor._methods import ravel_multi_range
from mlbol.dtensor._methods import unravel_index
from mlbol.dtensor._methods import moveaxis
from mlbol.dtensor._methods import swapaxes
from mlbol.dtensor._methods import roll
from mlbol.dtensor._methods import concatenate
from mlbol.dtensor._methods import stack
from mlbol.dtensor._methods import tile
from mlbol.dtensor._methods import flip
from mlbol.dtensor._methods import pad
from mlbol.dtensor._methods import broadcast_to
from mlbol.dtensor._methods import sort
from mlbol.dtensor._methods import count_nonzero
from mlbol.dtensor._methods import all
from mlbol.dtensor._methods import any
from mlbol.dtensor._methods import where
from mlbol.dtensor._methods import nonzero
from mlbol.dtensor._methods import sign
from mlbol.dtensor._methods import abs
from mlbol.dtensor._methods import conj
from mlbol.dtensor._methods import ceil
from mlbol.dtensor._methods import floor
from mlbol.dtensor._methods import round
from mlbol.dtensor._methods import square
from mlbol.dtensor._methods import sqrt
from mlbol.dtensor._methods import exp
from mlbol.dtensor._methods import log
from mlbol.dtensor._methods import log2
from mlbol.dtensor._methods import log10
from mlbol.dtensor._methods import sin
from mlbol.dtensor._methods import cos
from mlbol.dtensor._methods import tan
from mlbol.dtensor._methods import sinh
from mlbol.dtensor._methods import cosh
from mlbol.dtensor._methods import tanh
from mlbol.dtensor._methods import arcsin
from mlbol.dtensor._methods import arccos
from mlbol.dtensor._methods import arctan
from mlbol.dtensor._methods import arctan2
from mlbol.dtensor._methods import arcsinh
from mlbol.dtensor._methods import arccosh
from mlbol.dtensor._methods import arctanh
from mlbol.dtensor._methods import elu
from mlbol.dtensor._methods import relu
from mlbol.dtensor._methods import gelu
from mlbol.dtensor._methods import selu
from mlbol.dtensor._methods import sigmoid
from mlbol.dtensor._methods import silu
from mlbol.dtensor._methods import clip
from mlbol.dtensor._methods import maximum
from mlbol.dtensor._methods import minimum
from mlbol.dtensor._methods import max
from mlbol.dtensor._methods import min
from mlbol.dtensor._methods import sum
from mlbol.dtensor._methods import prod
from mlbol.dtensor._methods import mean
from mlbol.dtensor._methods import cumsum
from mlbol.dtensor._methods import cumprod
from mlbol.dtensor._methods import convolve_along_axis
from mlbol.dtensor._methods import moving_mean
from mlbol.dtensor._methods import eps
from mlbol.dtensor._methods import finfo
from mlbol.dtensor._methods import norm
from mlbol.dtensor._methods import logical_and
from mlbol.dtensor._methods import logical_or
from mlbol.dtensor._methods import add
from mlbol.dtensor._methods import subtract
from mlbol.dtensor._methods import multiply
from mlbol.dtensor._methods import divide
from mlbol.dtensor._methods import dot
from mlbol.dtensor._methods import matmul
from mlbol.dtensor._methods import tensordot
from mlbol.dtensor._methods import einsum
from mlbol.dtensor._methods import kron
from mlbol.dtensor._methods import gkron
from mlbol.dtensor._methods import qr
from mlbol.dtensor._methods import svd
from mlbol.dtensor._methods import eig
from mlbol.dtensor._methods import solve
from mlbol.dtensor._methods import lstsq
from mlbol.dtensor._methods import vectorize
from mlbol.dtensor._methods import isclose
from mlbol.dtensor._methods import jacobian
from mlbol.dtensor._methods import hessian
from mlbol.dtensor._methods import assert_allclose


_dtensor_engine.initialize()
_dtensor_engine.dispatch("A", target=sys.modules[__name__], is_static=True)
_dtensor_engine.dispatch("M", target=None, is_static=False)


def set_dtensor_backend(backend_name: str, threadsafe: bool = False) -> None:
    """Set backend for dense tensors.

    One must manually dispatch attributes in static mode to this module.

    Parameters
    ----------
    backend: {'numpy', 'pytorch'}
        Name of the backend to load.
    threadsafe : bool, optional, default is False
        If False, set the backend as default for all threads.
    """
    _dtensor_engine.set_backend(backend_name, threadsafe=threadsafe)
    _dtensor_engine.dispatch("A", target=sys.modules[__name__], is_static=True)


def get_dtensor_backend() -> str:
    """Get the name of backend for dense tensor.

    Returns
    -------
    str
        Name of backend.
    """
    return _dtensor_engine.get_backend_name()


@contextlib.contextmanager
def dtensor_profile(backend_name: str, threadsafe: bool = False) -> Generator:
    """Context manager to set the backend registry.

    Parameters
    ----------
    backend_name: {'numpy', 'pytorch'}
        The name of the backend to use. Default is 'numpy'.
    threadsafe : bool, optional
        If True, the backend will not become the default backend for all threads.
        Note that this only affects threads where the backend hasn't already
        been explicitly set. If False (default) the backend is set for the
        entire session.
    """
    old_backend_name = _dtensor_engine.get_backend_name()
    set_dtensor_backend(backend_name, threadsafe=threadsafe)
    try:
        yield
    finally:
        set_dtensor_backend(old_backend_name, threadsafe=threadsafe)
