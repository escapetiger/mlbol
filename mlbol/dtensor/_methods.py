import numpy as np
from typing import Any, Sequence, Callable, Iterable, Tuple, TypeVar
from mlbol.dtensor._engine import _DTensorEngine as _dtensor_engine
from mlbol.dtensor._attributes import Tensor

_ShapeLike = TypeVar("_ShapeLike")
_ScalarLike = TypeVar("_ScalarLike")
_TensorLike = TypeVar("_TensorLike")
_IndexLike = TypeVar("_IndexLike")


def _apply_method(name: str, *args, **kwargs) -> Any:
    for interface in [_dtensor_engine]:
        if hasattr(interface, name):
            return getattr(interface, name)(*args, **kwargs)
    raise ValueError(f"Method {name} is not available.")


def _make_tuple(x: _ShapeLike | None) -> Tuple[int, ...]:
    if x is None or isinstance(x, tuple):
        return x
    return tuple(x) if isinstance(x, Iterable) else (x,)


# ---- Default Context ----
def get_default_tensor_context() -> dict[str, Any]:
    """Get the default tensor context dictionary."""
    return _apply_method("get_default_tensor_context")


def set_default_tensor_context(**context) -> None:
    """Set the default tensor context dictionary."""
    return _apply_method("set_default_tensor_context", **context)


def get_tensor_dtype(name: str) -> Any:
    """Get tensor type from name."""
    return _apply_method("get_tensor_dtype", name)


def get_tensor_device(name: str) -> Any:
    """Get tensor device from name."""
    return _apply_method("get_tensor_device", name)


# ---- Tensor Creation ----
def is_tensor(obj: Any) -> bool:
    """Return if `obj` is a tensor for the current backend.

    Parameters
    ----------
    obj : Any
        Input data.

    Returns
    -------
    bool
        Boolean value indicating whether `obj` is a tensor
    """
    return _apply_method("is_tensor", obj)


def context(tensor: Tensor) -> dict[str, Any]:
    """Return the context of a tensor, i.e., a dictionary of the parameters
    characterising the tensor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.

    Returns
    -------
    dict[str, Any]
        Context dictionary of the input tensor.
    """
    return _apply_method("context", tensor)


def tensor(data: _TensorLike, **context) -> Tensor:
    """Return a tensor on the specified context, depending on the backend.

    Parameters
    ----------
    data : Any
        Input tensor-like data.

    Returns
    -------
    Tensor
        Output tensor.
    """
    if not context:
        context = get_default_tensor_context()
    return _apply_method("tensor", data, **context)


def as_tensor(data: _TensorLike, **context) -> Tensor:
    """Convert data to be a tensor based on the context. Copy data if necessary.

    Parameters
    ----------
    data : Any
        Input tensor-like data.

    Returns
    -------
    Tensor
        Output tensor.
    """

    if not context:
        context = get_default_tensor_context()
    return _apply_method("as_tensor", data, **context)


def to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a tensor to a NumPy array.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.

    Returns
    -------
    np.ndarray
        Numpy array.
    """
    return _apply_method("to_numpy", tensor)


def copy(tensor: Tensor) -> Tensor:
    """Return a copy of the given tensor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to be cloned.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("copy", tensor)


def empty(shape: _ShapeLike, **context) -> Tensor:
    """Return a new empty tensor with the given shape and context.

    Parameters
    ----------
    shape : int or sequence of ints
        Tensor shape.

    Returns
    -------
    Tensor
        Output tensor.
    """
    shape = _make_tuple(shape)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("empty", shape, **context)


def empty_like(tensor: Tensor):
    """Return a new empty tensor with the same shape and context as a given tensor.

    Parameters
    ----------
    tensor: Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("empty_like", tensor)


def zeros(shape: _ShapeLike, **context) -> Tensor:
    """Return a new tensor of the given shape and context, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Tensor shape.

    Returns
    -------
    Tensor
        Output tensor.
    """
    shape = _make_tuple(shape)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("zeros", shape, **context)


def zeros_like(tensor: Tensor):
    """Return a new tensor of zeros with the same shape and context as a given tensor.

    Parameters
    ----------
    tensor: Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("zeros_like", tensor)


def ones(shape: _ShapeLike, **context) -> Tensor:
    """Return a new tensor of the given shape and context, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Tensor shape.

    Returns
    -------
    Tensor
        Output tensor.
    """
    shape = _make_tuple(shape)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("ones", shape, **context)


def ones_like(tensor: Tensor) -> Tensor:
    """Return a new tensor of ones with the same shape and context as a given tensor.

    Parameters
    ----------
    tensor: Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("ones_like", tensor)


def full(shape: _ShapeLike, fill_value: _ScalarLike, **context) -> Tensor:
    """Return a new tensor of the given shape and context, filled with `filled_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Tensor shape.

    Returns
    -------
    Tensor
        Output tensor.
    """
    shape = _make_tuple(shape)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("full", shape, fill_value, **context)


def full_like(tensor: Tensor, fill_value: _ScalarLike) -> Tensor:
    """Return a full tensor with the same shape and context as a given tensor.

    Parameters
    ----------
    tensor: Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("full_like", tensor, fill_value)


def eye(N: int, M: int | None = None, **context) -> Tensor:
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output. If None, defaults to N.

    Returns
    -------
    Tensor
        Output tensor.
    """
    if not context:
        context = get_default_tensor_context()
    return _apply_method("eye", N, M=M, **context)


def diag(v: Tensor, k: int = 0, **context) -> Tensor:
    """Return a 2D tensor with the elements of `v` on the diagonal and zeros elsewhere.

    Parameters
    ----------
    v : Tensor
        Diagonal elements of the 2D tensor to construct. Required to be a 1D tensor.
    k : int, optional
        Index of diagonal: :math:`k=-1` for sub-diagonal and :math:`k=1` for sup-diagonal.
        By default 0.

    Returns
    -------
    Tensor
        A 2D diagonal tensor.
    """
    if not context:
        context = get_default_tensor_context()
    return _apply_method("diag", v, k=k, **context)


def diagonal(tensor: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Return specified diagonals.

    If `tensor` is 2D, returns the diagonal of `tensor` with the given offset, i.e.,
    the collection of elements of the form `tensor[i, i+offset]`. If `tensor` has
    more than two dimensions, then the axes specified by `axis1` and `axis2` are
    used to determine the 2D sub-tensor whose diagonal is returned. The shape of the
    resulting tensor can be determined by removing `axis1` and `axis2` and appending
    an index to the right equal to the size of the resulting diagonals.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    offset : int, optional
        Offset of the digaonal from the main diagonal. Can be positve (upper) or negative
        (lower). Default to main diagonal (0).
    axis1 : int, optional
        Axis to be used as the first axis of the 2D sub-tensors from which the diagonals
        should be taken. Default to first axis (0).
    axis2 : int, optional
        Axis to be used as the second axis of the 2D sub-tensors from which the diagonals
        should be taken. Default to second axis (1).

    Returns
    -------
    Tensor
        If `tensor` is 2D, returns a 1D tensor. If `tensor.ndim > 2`, then the dimensions
        specified by removing `axis1` and `axis2`, and insert a new axis at the end
        corresponding to the diagonal.
    """
    return _apply_method("diagonal", tensor, offset=offset, axis1=axis1, axis2=axis2)


def arange(
    start: _ScalarLike = 0,
    stop: _ScalarLike | None = None,
    step: _ScalarLike | None = None,
    **context,
) -> Tensor:
    """Return evenly spaced values within a specific interval under a given context.

    * If `step` is None, step size is set to be one.
    * If `stop` is None, return evenly spaced values within [0,start).
    * If `stop` is not None, return evenly spaced values within [start, stop).

    Parameters
    ----------
    start : scalar_like, optional
        Start value. By default 0.
    stop : scalar_like | None, optional
        Stop value. By default None.
    step : scalar_like | None, optional
        Step size. By default None.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("arange", start=start, stop=stop, step=step, **context)


def linspace(
    start: _ScalarLike, stop: _ScalarLike, num: int, endpoint: bool = True, **context
) -> Tensor:
    """Return evenly spaced values over a specific interval under a given context.

    * If `endpoint` is True, return `num` evenly spaced values within [start, stop].
    * If `endpoint` is False, return `num` evenly spaced values within [start, stop).

    Parameters
    ----------
    start : scalar_like
        Start value.
    stop : scalar_like
        Stop value.
    num : int
        Number of values.
    endpoint : bool, optional
        Whether include the right endpoint of a interval.

    Returns
    -------
    Tensor
        Output tensor.
    """
    if not context:
        context = get_default_tensor_context()
    return _apply_method("linspace", start, stop, num, endpoint=endpoint, **context)


def meshgrid(*tensors, indexing: str = "ij", **context) -> list[Tensor]:
    """Returns a list of coordinate matrices from coordinate vectors.

    Parameters
    ----------
    tensors : sequence of Tensor
        1D tensors representting the coordinates of a grid.
    indexing : str, optional
        The indexing mode, either "xy" or "ij", defaults to "ij".

        If "xy" is selected, the first dimension corresponds to the cardinality of the second input and the second dimension corresponds to the cardinality of the first input.

        If "ij" is selected, the dimensions are in the same order as the cardinality of the inputs.

    Returns
    -------
    list[Tensor]
    """
    if not context:
        context = get_default_tensor_context()
    return _apply_method("meshgrid", *tensors, indexing=indexing, **context)


def rand(shape: _ShapeLike, seed: int | None = None, **context) -> Tensor:
    """Returns a random tensor with samples in [0,1] from the uniform distribution.

    Parameters
    ----------
    shape : int or sequence of ints
        Tensor shape.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Output Tensor.
    """
    shape = _make_tuple(shape)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("rand", shape, seed=seed, **context)


def randn(shape: _ShapeLike, seed: int | None = None, **context) -> Tensor:
    """Returns a random tensor with samples from the standard normal distribution.

    Parameters
    ----------
    shape : int or sequence of ints
        Tensor shape.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Output Tensor.
    """
    shape = _make_tuple(shape)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("randn", shape, seed=seed, **context)


def gamma(
    shape: float | _TensorLike,
    scale: float | _TensorLike = 1.0,
    size: _ShapeLike | None = None,
    seed: int | None = None,
    **context,
) -> Tensor:
    """Draw samples from a Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    shape (sometimes designated .:math:`k`) and scale (sometimes designated :math:`\theta`),
    where both parameters are positive.

    Parameters
    ----------
    shape : float or tensor_like of floats
        The shape of the gamma distribution. Must be non-negative.
    scale : float or tensor_like of floats, optional
        The scale of the gamma dirtibution. Must be non-negative. Defaults to 1.
    size : int or sequence of ints, optional
        Output shape. If None (default), a single value is returned if `shape` and `scale` are both scalars.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Output Tensor.
    """
    size = _make_tuple(size)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("gamma", shape, scale=scale, size=size, seed=seed, **context)


def choice(
    t: _TensorLike | int,
    size: _ShapeLike | None = None,
    replace: bool = True,
    p: _TensorLike | None = None,
    seed: int | None = None,
    **context,
) -> Tensor:
    """Generate a random sample from a given 1D tensor.

    Parameters
    ----------
    t : tensor_like or int
        If a tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if it were `dl.arange(t)`.
    size : int or sequence of ints or None, optional
        Output shape. Default is None, in which  case a single value is returned.
    replace : bool, optional
        Whether the sample is with or without replacement. Default is True, meaning
        that a value of `t` can be selected multiple times.
    p : tensor_like or None, optional
        The probabilities associated with each entry in `t`. If None, the sample
        assumes a uniform distribution over all entries in `t`.
    seed : int | None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Output tensor.
    """
    size = _make_tuple(size)
    if not context:
        context = get_default_tensor_context()
    return _apply_method(
        "choice", t, size=size, replace=replace, p=p, seed=seed, **context
    )


def randint(
    low: int,
    high: int | None = None,
    size: _ShapeLike | None = None,
    seed: int | None = None,
    **context,
) -> Tensor:
    """Return random integers from `low` (inclusive) to `high` (exclusive).

    * If `high` is not None, the results are from :math:`[low, high)`.
    * If `high` is None (the default), then results are from :math:`[0,low)`.

    Parameters
    ----------
    low : int
        Lowest integer to be drawn from the distribution unless `high` is None.
    high : int or None, optional
        Highest integer. Dedault is None.
    size : int or sequence of ints or None, optional
        Output shape. Default is None.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Output tensor.
    """
    size = _make_tuple(size)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("randint", low, high=high, size=size, seed=seed, **context)


def permutation(x: int | _TensorLike, seed: int | None = None, **context) -> Tensor:
    """Randomly permute a sequence, or return a permuted range.

    If `x` is a multi-dimensional array, it is only shuffled along its first index.

    Parameters
    ----------
    x : int or tensor_like
        If an integer, randomly permute `dl.arange(x)`.
        If a tensor, make a copy and shuffle the elements randomly.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Output tensor.
    """
    if not context:
        context = get_default_tensor_context()
    return _apply_method("permutation", x, seed=seed, **context)


def uniform(
    low: float | tuple[float] = 0.0,
    high: float | tuple[float] = 1.0,
    size: int | tuple[int] | None = None,
    seed: int | None = None,
    **context,
) -> Tensor:
    """Randomly draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval `[low, high)`.

    Parameters
    ----------
    low : float or a tuple of floats, optional
        Lower boundary of the output interval. The default value is 0.
    high : float or a tuple of floats, optional
        Upper boundary of the output interval. The default value is 1.
    size : int or a tuple of ints or None, optional
        Output shape. If None (default), a single value is returned if
        `low` and `high` are both scalars. Otherwise, broadcast is applied.
        If the given shape is `(m, n, k)`, then `m * n * k` samples are drawn.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor or scalar
        Drawn samples from the parametrized unifrom distribution.
    """
    size = _make_tuple(size)
    if not context:
        context = get_default_tensor_context()
    return _apply_method("uniform", low=low, high=high, size=size, seed=seed, **context)


def normal(
    loc: float | tuple[float] = 0.0,
    scale: float | tuple[float] = 1.0,
    size: int | tuple[int] | None = None,
    seed: int | None = None,
    **context,
) -> Tensor:
    """Randomly draw samples from a normal distribution.

    Parameters
    ----------
    loc : float or a tuple of floats, optional
        Mean of the distribution. The deafult value is 0.
    scale : float or a tuple of floats, optional
        Standard deviation of the distribution. Must be non-negative. The default value is 1.
    size : int or a tuple of ints or None, optional
        Output shape. If None (default), a single value is returned if
        `low` and `high` are both scalars. Otherwise, broadcast is applied.
        If the given shape is `(m, n, k)`, then `m * n * k` samples are drawn.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor or scalar
        Drawn samples from the parametrized normal distribution.
    """
    size = _make_tuple(size)
    if not context:
        context = get_default_tensor_context()
    return _apply_method(
        "normal", loc=loc, scale=scale, size=size, seed=seed, **context
    )


def glorot_uniform(
    shape: _ShapeLike,
    scale: float = 1.0,
    mode: str = "fan_avg",
    seed: int | None = None,
    **context,
) -> Tensor:
    """Randomly draw samples from the Glorot uniform distribution.

    Samples are drawn from a uniform distribution on the interval :math:`[-limit, limit]`,
    with `limit = sqrt(3 * scale / n)`, where `n` is:

    * number of input units in the weight tensor, if mode = "fan_in";
    * number of output units, if mode = "fan_out";
    * average of the number of input and output units, if mode = "fan_avg";

    Parameters
    ----------
    shape : _ShapeLike
        Output shape.
    scale : float, optional
        Scaling factor. Must be positive. Default is 1.
    mode : {'fan_in', 'fan_out', 'fan_avg'}, optional
        Method to determine the parameters `n`. Default is 'fan_avg'.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Drawn samples from the Glorot uniform distribution.

    References
    ----------

    .. [1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.
    """
    shape = _make_tuple(shape)
    if not context:
        context = get_default_tensor_context()
    return _apply_method(
        "glorot_uniform", shape=shape, scale=scale, mode=mode, seed=seed, **context
    )


def glorot_normal(
    shape: _ShapeLike,
    scale: float = 1.0,
    mode: str = "fan_avg",
    truncated: bool = False,
    seed: int | None = None,
    **context,
) -> Tensor:
    """Randomly draw samples from the Glorot (truncated) normal distribution.

    Samples are drawn from a (truncated) normal distribution with a mean of zero and
    a standard deviation `stddev = sqrt(scale / n)`, where `n` is:

    * number of input units in the weight tensor, if mode = "fan_in";
    * number of output units, if mode = "fan_out";
    * average of the number of input and output units, if mode = "fan_avg";

    Parameters
    ----------
    shape : a sequence of ints
        Output shape.
    scale : float, optional
        Scaling factor. Must be positive. Default is 1.
    mode : {'fan_in', 'fan_out', 'fan_avg'}, optional
        Method to determine the parameters `n`. Default is 'fan_avg'.
    truncated : bool, optional
        If True, truncated normal distribution is used. Default is False.
    seed : int or None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Drawn samples from the Glorot uniform distribution.

    References
    ----------

    .. [1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2010.
    """
    shape = _make_tuple(shape)
    if not context:
        context = get_default_tensor_context()
    return _apply_method(
        "glorot_normal",
        shape=shape,
        scale=scale,
        mode=mode,
        truncated=truncated,
        seed=seed,
        **context,
    )


def exponential(
    scale: float | tuple[float] = 1.0,
    size: int | tuple[int] | None = None,
    seed: int | None = None,
    **context,
) -> Tensor:
    """Draw samples from an exponential distribution.

    If probability density function is

    .. math::

        f(x;\\frac{1}{\\beta}) = \\frac{1}{\\beta}\\exp(-\\frac{x}{\\beta}),

    for :math:`x>0` and 0 elsewhere. :math:`\\beta` is the scale paramter,
    which is the inverse of the rate parameter :math:`\\lambda`.

    Parameters
    ----------
    scale : float | tuple[float], optional
        The scale parameter. Must be non-negative.
    size : int | tuple[int] | None, optional
        Output shape.
    seed : int | None, optional
        Random seed. If None (default), NumPy's global seed is used.

    Returns
    -------
    Tensor
        Drawn samples from the exponential distribution.
    """
    return _apply_method("exponential", scale=scale, size=size, seed=seed, **context)


# ---- Tensor Manipulation ----
def size(tensor: Tensor, axis: int | None = None) -> int:
    """Return the number of elements along a given axis.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    axis : int or None, optional
        If None (default), return the total number of elements.

    Returns
    -------
    int
        The number of elements along `axis`.
    """
    return _apply_method("size", tensor, axis=axis)


def shape(tensor: Tensor) -> tuple[int]:
    """Returns the shape of the input tensor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.

    Returns
    -------
    tuple[int]
        Tensor shape.
    """
    return _apply_method("shape", tensor)


def ndim(tensor: Tensor) -> int:
    """Returns the number of dimensions of the input tensor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.

    Returns
    -------
    int
        Number of dimensions.
    """
    return _apply_method("ndim", tensor)


def index_update(tensor: Tensor, indices: _IndexLike, values: Tensor) -> Tensor:
    """Updates the value of tensors in the specified indices.

    Should be used as ::

            index_update(tensor, index[:, 3:5], values)

    Equivalent of ::

            tensor[:, 3:5] = values

    Parameters
    ----------
    tensor : Tensor
        Intput tensor which values to update.
    indices : index_like
        Indices to update.
    values : Tensor
        Values to use to fill tensor[indices].

    Returns
    -------
    Tensor
        Updated tensor.
    """
    return _apply_method("index_update", tensor, indices, values)


def reshape(tensor: Tensor, newshape: _ShapeLike) -> Tensor:
    """Gives a new shape to a tensor without changing its data.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    newshape : int or sequence of ints
        The new shape should be compatible with the original shape.
        If an integer, then the result will be a 1-D tensor of that length.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("reshape", tensor, newshape)


def transpose(tensor: Tensor, axes: tuple[int] | None = None) -> Tensor:
    """Returns a tensor with axes transposed.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    axes : tuple[int] or None, optional
        If None (default), it is `range(tensor.ndim)[::-1]`, which reverses
        the order of the axes. If not None, it must be a tuple which contains
        a permutation of [0,1,...,N-1] where N is the number of axes.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("transpose", tensor, axes=axes)


def ravel(tensor: Tensor) -> Tensor:
    """Return a contiguous flattened tensor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("ravel", tensor)


def ravel_multi_index(
    multi_index: tuple[Tensor], shape: _ShapeLike, mode: str = "raise", order: str = "C"
) -> Tensor:
    """Convert a tuple of index tensors into a tensor of flat indices, applying boundary modes
    to the multi-index.

    Parameters
    ----------
    multi_index : tuple of Tensors
        A tuple of integer tensors, one tensor for each dimension.
    shape : shape_like
        The shape of tensor into which the indices from `multi_index` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled. Can spcify either one mode or a tuple
        of modes, one mode per index.
            - 'raise' - raise an error (default)
            - 'wrap' - wrap around
            - 'clip' - clip to the range
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as indexing in row-major (C-style) or
        column-major (Fortran-style) order.

    Returns
    -------
    Tensor
        A tensor of indices into the flattened version of a tensor of dimensions `shape`.
    """
    shape = _make_tuple(shape)
    return _apply_method(
        "ravel_multi_index", multi_index, shape, mode=mode, order=order
    )


def ravel_multi_range(
    low: _ShapeLike,
    high: _ShapeLike,
    shape: _ShapeLike,
    mode: str = "raise",
    order: str = "C",
) -> Tensor:
    """Generate a tensor of flat indices from a tuple of ranges, applying boundary modes to the
    multi-index.

    Parameters
    ----------
    low : tuple of ints
        Lower bounds of ranges.
    high : tuple of ints
        Upper bounds of ranges.
    shape : tuple of ints
        The shape of tensor into which the indices from `multi_index` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled. Can spcify either one mode or a tuple
        of modes, one mode per index.
            - 'raise' - raise an error (default)
            - 'wrap' - wrap around
            - 'clip' - clip to the range
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as indexing in row-major (C-style) or
        column-major (Fortran-style) order.

    Returns
    -------
    Tensor
        A tensor of indices into the flattened version of a tensor of dimensions `n`.
    """
    low = _make_tuple(low)
    high = _make_tuple(high)
    shape = _make_tuple(shape)
    return _apply_method("ravel_multi_range", low, high, shape, mode=mode, order=order)


def unravel_index(
    indices: _TensorLike, shape: _ShapeLike, order: str = "C"
) -> tuple[Tensor]:
    """Convert a flat index or tensor of indices into a tuple of coordinate tensors.

    Parameters
    ----------
    indices : tensor_like
        An integer tensor whose elements are indices into the flattened version of a tensor of
        dimensions `shape`.
    shape : shape_like
        The shape of the tensor to use for unraveling `indices`.
    order : {'C', 'F'}, optional
        Determine whether the indices should be viewed as indexing in row-major (C-style) or
        column-major (Fortran-style) order. Default is 'C'.

    Returns
    -------
    tuple of Tensors
        Each tensor in the tuple has the same shape as the `indices` tensor.
    """
    shape = _make_tuple(shape)
    return _apply_method("unravel_index", indices, shape, order=order)


def moveaxis(
    tensor: Tensor, src: int | Sequence[int], dst: int | Sequence[int]
) -> Tensor:
    """Move axes of a tensor to new positions, while other axes remain
    in their original order.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    src : int or sequence of int
        Original positions of the axes to move. These must be unique.
    dst : int
        Destination positions for each of the original axes. These must also be unique.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("moveaxis", tensor, src, dst)


def swapaxes(tensor: Tensor, axis1: int, axis2: int) -> Tensor:
    """Interchange two axes of a tensor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("swapaxes", tensor, axis1, axis2)


def roll(
    tensor: Tensor, shift: int | tuple[int], axis: int | tuple[int] | None = None
) -> Tensor:
    """Roll tensor elements along a given axis.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    shift : int or tuple of ints
        The number of places by which elements are shifted. If a tuple, then `axis`
        must be a tuple of the same size, and each of the given axes is shifted by
        the corresponding number. If an int while `axis` is a tuple of ints, then
        the same value is used for all given axes.
    axis : int | None, optional
        Axis or axes along which elements are shifted. By default, the tensor is
        flattened before shifting, after which the original shape is restored.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("roll", tensor, shift, axis)


def concatenate(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Join a sequence of tensors along an exisiting axis.

    Parameters
    ----------
    tensors : sequence of Tensor
        Non-empty tensors provided must have the same shape, except along the specified axis.
    axis : int, optional
        The axis along which the tensor swill be joined. Default is 0.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("concatenate", tensors, axis)


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Join a sequence of tensors along a new axis.

    Parameters
    ----------
    tensors : sequence of Tensor
        Each tensor must have the same shape.
    axis : int, optional
        The axis in the result tensor along which the input tensors are
        stacked.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("stack", tensors, axis)


def tile(tensor: Tensor, reps: int | tuple[int]) -> Tensor:
    """Contruct a new tensor by repeating a specific tensor the number of times given by `reps`.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    reps : int | tuple[int]
        The number of repetitions along each axis.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("tile", tensor, reps)


def flip(tensor: Tensor, axis: int | tuple[int] = 0) -> Tensor:
    """Reverse the order of elements in a tensor along the given axis.

    The shape of the tensor is preserved, but the elements are reordered.

    Notes
    -----
    * `numpy.flip` returns a view in constant time.
    * `torch.flip` makes a copy of input's data, hence slower than `numpy.flip`.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    axis : int or tuple of ints, optional
        Axis or axes along which to flip over. Default is 0.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("flip", tensor, axis)


def pad(
    tensor: Tensor,
    width: int | tuple[int] | Sequence[tuple[int]],
    mode: str = "constant",
    stat_length: int | Sequence[int] | None = None,
    constant_values: _ScalarLike | Sequence[_ScalarLike] | None = None,
    end_values: _ScalarLike | Sequence[_ScalarLike] = 0,
    reflect_type: str = "even",
) -> Tensor:
    """Pad a tensor with a specific style.

    This function always use `numpy.pad` as its backend.

    Notes
    -----
    * `numpy.pad` supports tensor with arbitrary number of dimensions.
    * `torch.pad` is more restricted. Constant padding supports for arbitrary
        dimensions. Circular, replicate and reflection padding are implemented
        for padding the last 3 dimensions of a 4D or 5D tensor, the last 2
        dimension of a 3D or 4D tensor, or the last dimension of a 2D or 3D
        tensor. In implementation, it'd better to unsqueeze a low-dimensional
        tensor to a high-dimensional tensor, and then call `torch.pad`.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    width : int, or tuple of ints, or sequence of tuples of ints
        Number of values padded to the edges of each axis. Pad width is of form
        :math:`(before_1,after_1),...,(before_N,after_N)`, where :math:`N` is
        the dimension of input tensor. :math:`(pad,)` and `int` is a shortcut for
        before = after = pad width for all axes.
    mode : str, optional
        One of the following string values.
        * "constant" (default): pads with a constant value.
        * "edge": pads the edge values of tensor.
        * "linear_ramp": Pads with the linear ramp between `end_value` and the
            tensor edge value.
        * "maximum": Pads with the maximum value of all or part of the vector
            along axis.
        * "minimum": Pads with the minimum value of all or part of the vector
            along axis.
        * "median": Pads with the median value of all or part of the vector
            along axis.
        * "wrap": pads with the wrap of the vector along the axis.
        * "reflect": pads with the reflection of the vector mirrored on
            the first and last values of the vector along each axis.
    stat_length : sequence or int, optional
        Used in "maximum", "mean", "median", and "minimum". Number of values at
        edge of each axis used to calculate the statistic value. Default is None,
        to use the entire axis.
    constant_values : sequence or scalar, optional
        Used in "constant", default is None. Must be a scalar.
    end_values : sequence or scalar, optional
        Used in "linear_ramp". The values used for the ending value of the
        linear_ramp and that will form the edge of the padded tensor. Default is 0.
    reflect_type : {"even", "odd"}, optional.
        Used in "reflect". The "even" style is the default with an unaltered
        reflection around the edge value. For the "odd" style, the extended
        part of the tensor is creted by subtracting the reflected values from two times
        the edge value.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method(
        "pad",
        tensor,
        width,
        mode=mode,
        stat_length=stat_length,
        constant_values=constant_values,
        end_values=end_values,
        reflect_type=reflect_type,
    )


def broadcast_to(tensor: Tensor, shape: tuple[int]) -> Tensor:
    """Broadcast a tensor to a new shape.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    shape : a tuple of ints
        The shape of the desired tensor.

    Returns
    -------
    Tensor
        A readonly view on the original tensor with the given shape. It is
        typically not contiguous. Furthermore, more than one element of a
        broadcasted tensor may refere to a single memory location.
    """
    return _apply_method("broadcast_to", tensor, shape)


# ---- Sorting and Searching ----
def sort(tensor: Tensor, axis: int = -1) -> Tensor:
    """Return a sorted copy of a tensor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    axis : int, optional
        Axis along which to sort. The default is -1, which sorts along the last axis.

    Returns
    -------
    Tensor
        Output tensor
    """
    return _apply_method("sort", tensor, axis=axis)


def count_nonzero(tensor: Tensor, axis: int = None) -> Tensor:
    """Count the number of non-zero values in a specific tensor,
    optionally along an axis.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    axis : int, optional
        If None (default), all non-zeros will be counted along a flattend tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("count_nonzero", tensor, axis=axis)


def all(tensor: Tensor, axis: int | tuple[int] = None) -> Tensor:
    """Test whether all tensor elements along a given axis evaluate to True.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    axis : int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (`axis=None`) is to perform a logical AND over all
        the dimensions of the input tensor. If this is a tuple of ints,
        a reduction is performed on multiple axes, instead of a single
        axis or all the axes as before.

    Returns
    -------
    Tensor
        A new boolean or tensor is returned.
    """
    return _apply_method("all", tensor, axis=axis)


def any(tensor: Tensor, axis: int | tuple[int] = None) -> Tensor:
    """Test whether any tensor elements along a given axis evaluate to True.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    axis : int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (`axis=None`) is to perform a logical OR over all
        the dimensions of the input tensor. If this is a tuple of ints,
        a reduction is performed on multiple axes, instead of a single
        axis or all the axes as before.

    Returns
    -------
    Tensor
        A new boolean or tensor is returned.
    """
    return _apply_method("any", tensor, axis=axis)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Return elements either from `x` or `y`, depending on `condition`.

    Parameters
    ----------
    condition : Tensor
        When True, yield element from `x`, otherwise from `y`.
    x, y : Tensor
        Values from which to choose.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("where", condition, x, y)


def nonzero(tensor: Tensor, as_tuple: bool = True) -> Tensor | tuple[Tensor]:
    """Returns the indices of the elements that are non-zero.

    In NumPy, it always returns a tuple of tensors, one for each dimension of `tensor`,
    containing the indices of the non-zero elements in that dimension.

    In PyTorch, when `as_tuple` is False, it returns a 2D Tensor with each column
    containing the indices of the non-zero elements in that dimension. When `as_tuple`
    is True, it is the same as in NumPy.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    as_tuple : bool, optional
        With NumPy, this must be True.
        With PyTorch, this option affects the type of outputs.

    Returns
    -------
    Tensor or a tuple of Tensors
        Indices of elements that are non-zero.
    """
    return _apply_method("nonzero", tensor, as_tuple=as_tuple)


# ---- Mathemetical Function: Part I ----
def sign(x: Tensor) -> Tensor:
    """Calculate the sign of all elements in the given input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("sign", x)


def abs(x: Tensor) -> Tensor:
    """Calculate the absolute value of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("abs", x)


def conj(x: Tensor) -> Tensor:
    """Return the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by
    changing the sign of its imaginary part.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("conj", x)


def ceil(x: Tensor) -> Tensor:
    """Return the ceiling of the input tensor, element-wise.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("ceil", x)


def floor(x: Tensor) -> Tensor:
    """Return the floor of the input tensor, element-wise.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("floor", x)


def round(x: Tensor, decimals: int = 0) -> Tensor:
    """Evenly round to the given number of decimals.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    decimals: int
        Default is 0.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("round", x, decimals=decimals)


def square(x: Tensor) -> Tensor:
    """Calculate the square of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("square", x)


def sqrt(x: Tensor) -> Tensor:
    """Calculate the square root of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("sqrt", x)


def exp(x: Tensor) -> Tensor:
    """Calculate the exponential of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("exp", x)


def log(x: Tensor) -> Tensor:
    """Calculate the natural logarithm of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("log", x)


def log2(x: Tensor) -> Tensor:
    """Calculate the base-2 logarithm of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("log2", x)


def log10(x: Tensor) -> Tensor:
    """Calculate the base-10 logarithm of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("log10", x)


def sin(x: Tensor) -> Tensor:
    """Calculate the sine of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("sin", x)


def cos(x: Tensor) -> Tensor:
    """Calculate the cosine of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("cos", x)


def tan(x: Tensor) -> Tensor:
    """Calculate the tangent of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("tan", x)


def sinh(x: Tensor) -> Tensor:
    """Calculate the hyperbolic sine of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("sinh", x)


def cosh(x: Tensor) -> Tensor:
    """Calculate the hyperbolic cosine of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("cosh", x)


def tanh(x: Tensor) -> Tensor:
    """Calculate the hyperbolic tangent of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("tanh", x)


def arcsin(x: Tensor) -> Tensor:
    """Calculate the inverse sine of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("arcsin", x)


def arccos(x: Tensor) -> Tensor:
    """Calculate the inverse cosine of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("arccos", x)


def arctan(x: Tensor) -> Tensor:
    """Calculate the inverse tangent of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("arctan", x)


def arctan2(x1: Tensor, x2: Tensor) -> Tensor:
    """Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    Parameters
    ----------
    x1 : Tensor
        y-coordinates.
    x2 : Tensor
        x-coordinates.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("arctan2", x1, x2)


def arcsinh(x: Tensor) -> Tensor:
    """Calculate the inverse hyperbolic sine of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("arcsinh", x)


def arccosh(x: Tensor) -> Tensor:
    """Calculate the inverse hyperbolic cosine of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("arccosh", x)


def arctanh(x: Tensor) -> Tensor:
    """Calculate the inverse hyperbolic tangent of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("arctanh", x)


def elu(x: Tensor) -> Tensor:
    """Calculate the exponential linear unit (ELU) of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("elu", x)


def relu(x: Tensor) -> Tensor:
    """Calculate the rectified linear unit (ReLU) of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("relu", x)


def gelu(x: Tensor) -> Tensor:
    """Calculate the Gaussian error linear unit (GELU) of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("gelu", x)


def selu(x: Tensor) -> Tensor:
    """Calculate the scaled exponential linear unit (SELU) of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("selu", x)


def sigmoid(x: Tensor) -> Tensor:
    """Calculate the sigmoid of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("sigmoid", x)


def silu(x: Tensor) -> Tensor:
    """Calculate the sigmoid linear unit (SiLU) of all elements in the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("silu", x)


def clip(
    x: Tensor, a_min: _ScalarLike | None = None, a_max: _ScalarLike | None = None
) -> Tensor:
    """Clip the values of a tensor to within an interval.

    Given an interval, values outside the interval are clipped to the interval
    edges.  For example, if an interval of ``[0, 1]`` is specified, values
    smaller than 0 become 0, and values larger than 1 become 1.

    Not more than one of `a_min` and `a_max` may be `None`.

    Parameters
    ----------
    x : tensor
        Input tensor.
    a_min : scalar, optional
        Minimum value. If `None`, clipping is not performed on lower bound.
    a_max : scalar, optional
        Maximum value. If `None`, clipping is not performed on upper bound.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("clip", x, a_min=a_min, a_max=a_max)


def maximum(x1: Tensor, x2: Tensor) -> Tensor:
    """Element-wise maximum of tensor elements.

    Parameters
    ----------
    x1, x2 : Tensor
        Two tensors to be compared.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("maximum", x1, x2)


def minimum(x1: Tensor, x2: Tensor):
    """Element-wise minimum of tensor elements.

    Parameters
    ----------
    x1, x2 : Tensor
        Two tensors to be compared.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("minimum", x1, x2)


# ---- Mathematical Function: Part II ----
def max(x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Find the maximum of the input tensor, optionally along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int or None, optional
        Axis along which to operate. Default is None.
    keepdims : bool, optional
        If this is set to True, the axis which is reduced is left in the result
        as a dimension with size one. Default is False.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("max", x, axis=axis, keepdims=keepdims)


def min(x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Find the minimum of the input tensor, optionally along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int or None, optional
        Axis along which to operate. Default is None.
    keepdims : bool, optional
        If this is set to True, the axis which is reduced is left in the result
        as a dimension with size one. Default is False.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("min", x, axis=axis, keepdims=keepdims)


def sum(x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Calculate the sum of the input tensor, optionally along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int or None, optional
        Axis along which to operate. Default is None.
    keepdims : bool, optional
        If this is set to True, the axis which is reduced is left in the result
        as a dimension with size one. Default is False.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("sum", x, axis=axis, keepdims=keepdims)


def prod(x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Calculate the product of the input tensor, optionally along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int or None, optional
        Axis along which to operate. Default is None.
    keepdims : bool, optional
        If this is set to True, the axis which is reduced is left in the result
        as a dimension with size one.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("prod", x, axis=axis, keepdims=keepdims)


def mean(x: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """Calculate the mean of the input tensor, optionally along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int or None, optional
        Axis along which to operate. Default is None.
    keepdims : bool, optional
        If this is set to True, the axis which is reduced is left in the result
        as a dimension with size one.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("mean", x, axis=axis, keepdims=keepdims)


def cumsum(x: Tensor, axis: int | None = None) -> Tensor:
    """Calculate the cumulative sum of the input tensor, optionally along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int or None, optional
        Axis along which to operate. Default is None.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("cumsum", x, axis=axis)


def cumprod(x: Tensor, axis: int | None = None) -> Tensor:
    """Calculate the cumulative product of the input tensor, optionally along an axis.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : int or None, optional
        Axis along which to operate. Default is None.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("cumprod", x, axis=axis)


def convolve_along_axis(x: Tensor, w: Tensor, axis: int = 0) -> Tensor:
    """Convolution of two tensors along an axis.

    Notes
    -----
    `torch.conv1d` is quite different from `numpy.convolve`. For consistency,
    we flip weight tensor first before call `torch.conv1d`.


    Parameters
    ----------
    x, w : Tensor
        Input tensors.
    axis : int, optional
        Axis along which to operate. Default is 0.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("convolve_along_axis", x, w, axis=axis)


def moving_mean(x: Tensor, n: int = 2, axis: int = 0) -> Tensor:
    """Calculate the moving mean of the input tensor along a given axis.

    Parameters
    ----------
    a : tensor
        Input tensor.
    n : int, optional
        Window size.
    axis : int, optional
        Axis along which to operate. Default is 0.

    Returns
    -------
    tensor
        The moving mean.
    """
    return _apply_method("moving_mean", x, n=n, axis=axis)


# ---- Linear Algebra ----
def eps(dtype: Any) -> Any:
    """Returns the machine epsilon for a given floating point dtype.

    Parameters
    ----------
    dtype : dtype
        The dtype for which to get the machine epsilon.

    Returns
    -------
    eps
        Machine epsilon for `dtype`
    """
    return _apply_method("eps", dtype)


def finfo(dtype: Any) -> Any:
    """Machine limits for floating point types.

    Parameters
    ----------
    dtype: float, dtype or instance
        Kind of floating point data-type about which to get information.
    """
    return _apply_method("finfo", dtype)


def norm(
    x: Tensor,
    ord: int | str = 2,
    axis: int | tuple[int] | None = None,
    keepdims: bool = False,
):
    """Computes the l-`order` norm of the input tensor, optionally along some axes.

    Parameters
    ----------
    x : tensor
        Input tensor.
    ord : {non-zero int, dl.inf, 'fro', 'nuc'}, optional
        Order of the norm. The default is 2.
    axis : int, tuple of ints or None, optional
        If `axis` is an integer, it specifies the axis of `x` along which to compute
        the vector norm. If `axis` is a 2-tuple, it specifies the axes that hold
        2D matrices, and the matrix norms of these matrices are computed. If `axis`
        is None then either a vector norm or a matrix norm is returned. Default is None.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the result
        as dimensions with size one. Default is False.

    Returns
    -------
    float or tensor
        Norm of the matrix or vector(s).
    """
    return _apply_method("norm", x, ord=ord, axis=axis, keepdims=keepdims)


def logical_and(x: Tensor, y: Tensor) -> Tensor:
    """Copmute the truth value of `x` AND `y` element-wise.

    Parameters
    ----------
    x, y : Tensor
        Input tensors.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("logical_and", x, y)


def logical_or(x: Tensor, y: Tensor) -> Tensor:
    """Copmute the truth value of `x` OR `y` element-wise.

    Parameters
    ----------
    x, y : Tensor
        Input tensors.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("logical_or", x, y)


def add(x: Tensor, y: Tensor) -> Tensor:
    """Calculate the elementwise addition between `x` and `y`,
    i.e., :math:`x+y`.

    Parameters
    ----------
    x, y : Tensor
        Input tensors.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("add", x, y)


def subtract(x: Tensor, y: Tensor) -> Tensor:
    """Calculate the elementwise subtract between `x` and `y`,
    i.e., :math:`x-y`.

    Parameters
    ----------
    x, y : Tensor
        Input tensors.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("subtract", x, y)


def multiply(x: Tensor, y: Tensor) -> Tensor:
    """Calculate the elementwise multiplication between `x` and `y`,
    i.e., :math:`x*y`.

    Parameters
    ----------
    x, y : Tensor
        Input tensors.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("multiply", x, y)


def divide(x: Tensor, y: Tensor) -> Tensor:
    """Calculate the elementwise division between `x` and `y`,
    i.e., :math:`x/y`.

    Parameters
    ----------
    x, y : Tensor
        Input tensors.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("divide", x, y)


def dot(x: Tensor, y: Tensor) -> Tensor:
    """Calculate the dot product between `x` and `y`,
    i.e., :math:`x \\cdot y`.

    Parameters
    ----------
    x, y : Tensor
        Input tensors.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("dot", x, y)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication of tensors representing (batches of) matrices.

    Parameters
    ----------
    a, b : tensor
        tensors representing the matrices to contract

    Returns
    -------
    a @ b
        matrix product of a and b

    Notes
    -----
    The behavior depends on the arguments in the following way.
        * If both arguments are 2-D they are multiplied like conventional matrices.
        * If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        * If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
        * If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.

    `matmul` differs from dot in two important ways:
        * Multiplication by scalars is not allowed, use * instead.
        * Stacks of matrices are broadcast together as if the matrices were elements, respecting the signature ``(n,k),(k,m)->(n,m)``:

        .. code-block:: python

            >>> a = np.ones([9, 5, 7, 4])

            >>> c = np.ones([9, 5, 4, 3])

            >>> np.dot(a, c).shape
            (9, 5, 7, 9, 5, 3)

            >>> np.matmul(a, c).shape
            (9, 5, 7, 3)

            >>> # n is 7, k is 4, m is 3

    The matmul function implements the semantics of the ``@`` operator introduced in Python 3.5 following `PEP 465 <https://www.python.org/dev/peps/pep-0465/>`_.
    """
    return _apply_method("matmul", a, b)


def tensordot(a: Tensor, b: Tensor, axes: int | tuple[int] = 2) -> Tensor:
    """Compute tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an tensor_like object containing
    two tensor_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : Tensor
        Tensors to "dot".
    axes : int or 2-tuple of ints, optional
        If an int N, sum over the last N axes of `a` and the first N axes
        of `b` in order. The sizes of the corresponding axes must match.
        If a 2-tuple of ints, a list of axes to be summed over, first sequence
        applying to `a`, second to `b`. Default is 2.

    Returns
    -------
    Tensor
        The tensor dot product of the input.

    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.
    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    The shape of the result consists of the non-contracted axes of the
    first tensor, followed by the non-contracted axes of the second.
    """
    return _apply_method("tensordot", a, b, axes=axes)


def einsum(subscripts: str, *operands) -> Tensor:
    """Evaluates the Einstein summation convention on the operands.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.

    *operands : list of Tensors
        tensors for the operation

    Returns
    -------
    Tensor
        The calculation based on the Einstein summation convention

    Notes
    -----
    This is only available for certain backends.
    """
    return _apply_method("einsum", subscripts, *operands)


def kron(a: Tensor, b: Tensor) -> Tensor:
    """Kronecker product of two 1D tensors.

    Parameters
    ----------
    a, b : Tensor
        Input tensors. Must be 1D tensors.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return _apply_method("kron", a, b)


def gkron(xs: list[Tensor]) -> list[Tensor]:
    """Generalized Kronecker product of a list of 2D tensors.

    Parameters
    ----------
    xs : list of Tensor
        Input tensors. Must be 2D tensors.

    Returns
    -------
    list[Tensor]
        Output tensors.
    """
    return _apply_method("gkron", xs)


def qr(A: Tensor) -> tuple[Tensor]:
    """Compute the qr factorization of a matrix.

    Factor the matrix `A` as *QR*, where `Q` is orthonormal and `R` is
    upper-triangular.

    Parameters
    ----------
    A : (..., M, N) Tensor
        Matrix to be factored.

    Returns
    -------
    Q : Tensor
        A matrix with orthogonal columns.
    R : Tensor
        The upper-traingular matrix.
    """
    return _apply_method("qr", A)


def svd(A: Tensor) -> tuple[Tensor]:
    """Compute the singular value decomposition (SVD) of a matrix.

    Parameters
    ----------
    A : (..., M, N) Tensor
        Matrix to be factored with `a.ndim >= 2`.

    Returns
    -------
    U : {(..., M, M), (..., M, K)} Tensor
        Unitary tensor.
    S : (..., K) Tensor
        Vector with the singular values, within each evector sorted in descending order.
    Vh : {(..., N, N), (..., K, N)} Tensor
        Unitary tensor.
    """
    return _apply_method("svd", A)


def eig(A: Tensor) -> tuple[Tensor]:
    """Compute the eigenvalues and right eigenvectors of a real-valued square matrix.

    Parameters
    ----------
    A : (..., M, N) Tensor
        Matrices for which the eigenvalues and right eigenvectors will be computed.

    Returns
    -------
    eigenvalues : (..., M) Tensor
        The eigenvalues, each repeated accroding to its multiplicity.
    eigenvectors : (..., M, M) Tensor
        The normalized eigenvectors, such that the i-th column is the eigenvector correponding
        to the i-th eigenvalue.
    """
    return _apply_method("eig", A)


def solve(A: Tensor, b: Tensor) -> Tensor:
    """Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, `x`, of the well-determined, i.e., full
    rank, linear matrix equation :math:`Ax = b`.

    Parameters
    ----------
    A : (..., M, M) Tensor
        Coefficient matrix.
    b : {(..., M), (..., M, K)} Tensor
        Ordinate values.

    Returns
    -------
    x : {(...,M), (..., M, K)} Tensor
        Solution to the system :math:`Ax = b`. Returned shape is identical to `b`.
    """
    return _apply_method("solve", A, b)


def lstsq(A: Tensor, b: Tensor) -> tuple[Tensor]:
    """Computes a solution to the least squares problem :math:`\\|Ax-b\\|_F`

    If the coefficient martix is underdetermined (m<n) and multiple
    solutions exist, the min norm solution is returned.

    Parameters
    ----------
    A : (..., M, N) Tensor
        Coefficient matrix.
    b : {(..., M), (..., M, K)} Tensor
        Ordinate values.

    Returns
    -------
    x : {(..., N), (..., N, K)} Tensor
        Solution to the least squares problem :math:`\\|Ax-b\\|_F`.
    residuals : (..., K) Tensor
        Sums of squared residuals: Squared Euclidean 2-norm for each column in Ax-b.
        If the rank of a is < N or M <= N, this is an empty tensor.
    """
    return _apply_method("lstsq", A, b)


def vectorize(func: Callable) -> Callable:
    """Vectorize a function."""
    return _apply_method("vectorize", func)


def isclose(
    a: _TensorLike,
    b: _TensorLike,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> _TensorLike:
    """Returns a new tensor with boolean elements representing if each element of `a`
    is close to the corresponding element of `b`. Clossness is defined as

    .. math::

        \\abs{a - b} \\le \\text{atol} + \\text{rtol} \\abs{b}

    Parameters
    ----------
    a, b : tensor_like,
        Input tensor to compare.
    rtol : float, optional
        The relative tolerance parameter.
    atol : float, optional
        The absolute tolerance parameter.
    equal_nan : bool, optional
        Whether to compare NaN's as equal.
        If True, NaN's in `a` will be considered equal to NaN's in `b`.

    Returns
    -------
    tensor_like
        Boolean tensor of where `a` and `b` are equal within the given tolerance.
    """
    return _apply_method("isclose", a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


# ---- Automatic Differentiation ----
def jacobian(
    ys: Tensor,
    xs: Tensor,
    i: int = 0,
    j: int | None = None,
    lazy: bool = False,
) -> Tensor:
    """Compute Jacobian matrix :math:`J` with :math:`J_{ij} = \frac{dy_i}{dx_j}`,
    where :math:`i = 0, ..., n_y-1` and :math:`j = 0, ..., n_x-1`.

    Parameters
    ----------
    ys : (..., Ny) Tensor
        Output tensor. Must be a 2D tensor.
    xs : (..., Nx) Tensor
        Input tensor. Must be a 2D tensor.
    i : int, optional
        Output index for which derivatives are computed. Default is zero.
    j : int or None, optional
        If None (default), all the derivatives related to i-th output are computed.
    lazy : bool, optional
        Whether enable lazy evaluation. Default is False.

    Returns
    -------
    Tensor
        Jacobian values :math:`J_{ij}`.
    """
    return _apply_method("jacobian", ys, xs, i=i, j=j, lazy=lazy)


def hessian(
    ys: Tensor,
    xs: Tensor,
    i: int | None = None,
    j: int = 0,
    k: int = 0,
    lazy: bool = False,
) -> Tensor:
    """
    Compute Hessian matrix H with :math:`H_{jk} = \frac{d^2y}{dx_jdx_k}`,
    where :math:`j,k = 0,..., n_x-1`.

    Notes
    -----
    There is some bugs in `hessian`!!!

    Parameters
    ----------
    ys : (..., Ny) Tensor
        Output tensor. Must be a 2D tensor.
    xs : (..., Nx) Tensor
        Input tensor. Must be a 2D tensor.
    i : int or None, optional
        If ny > 1, then `ys[:, component]` is used as y to compute the
        Hessian. If ny = 1, `i` must be ``None``. Default is None.
    j : int, optional
        Index of Input. Default is zero.
    k : int, optional
        Index of Output. Default is zero.
    lazy : bool, optional
        Whether enable lazy evaluation. Default is False.

    Returns
    -------
    Tensor
        Hessian values :math:`H_{ij}`.
    """
    return _apply_method("hessiand", ys, xs, i=i, j=j, k=k, lazy=lazy)


# # ---- File I/O ----
# def save_to_mat(file: str, data: Tensor) -> None:
#     """Save data to a binary file in '.mat' format.

#     Parameters
#     ----------
#     file : str
#         Name of output file.
#     data : Tensor
#         Data to be store.
#     """
#     return _apply_method("save_to_mat", file, data)


# def save_to_npy(file: str, data: Tensor) -> None:
#     """Save data to a binary file in '.mat' format.

#     Parameters
#     ----------
#     file : str
#         Name of output file.
#     data : Tensor
#         Data to be store.
#     """
#     return _apply_method("save_to_npy", file, data)


# def load_from_mat(file: str) -> Tensor:
#     """Load data from a binary file in '.mat' format.

#     Parameters
#     ----------
#     file: str
#         Name of input file.

#     Returns
#     -------
#     data : Tensor
#     """
#     return _apply_method(file)


# def load_from_npy(file: str) -> Tensor:
#     """Load data from a binary file in '.npy' format.

#     Parameters
#     ----------
#     file: str
#         Name of input file.

#     Returns
#     -------
#     data : Tensor
#     """
#     return _apply_method(file)


# ---- Unit Test ----
def assert_allclose(
    a: _TensorLike,
    b: _TensorLike,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    err_msg: str | None = None,
) -> None:
    """Raises an AssertionError if two objects are not equal up to desired tolerance.
    Clossness is defined as

    .. math::

        \\abs{a - b} \\le \\text{atol} + \\text{rtol} \\abs{b}.

    Parameters
    ----------
    a, b : tensor_like,
        Input tensor to compare.
    rtol : float, optional
        The relative tolerance parameter.
    atol : float, optional
        The absolute tolerance parameter.
    equal_nan : bool, optional
        Whether to compare NaN's as equal.
        If True, NaN's in `a` will be considered equal to NaN's in `b`.
    err_msg : str or None, optional
        Error messgae to print. Default is None.
    """
    return _apply_method(
        "assert_allclose",
        a,
        b,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
    )
