import math
import numpy as np
import itertools
from typing import Any, Dict
from mlbol.utils import GroupedRegistry
from mlbol.api.dtensor.utils import get_numpy_random_state, savemat, loadmat

backend = GroupedRegistry("numpy", group=("A", "M"))

default_context = {"dtype": np.float32}
dtype_dict = {
    "int32": np.int32,
    "int64": np.int64,
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
}


# ---- Attributes ----
@backend.register("A")
class Tensor(np.ndarray):
    @property
    def context(self) -> Dict[str, Any]:
        return {"dtype": self.dtype}


for name in [
    "int32",
    "int64",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "e",
    "pi",
    "inf",
    "nan",
]:
    backend.register_("A", name, getattr(np, name))


# ---- Methods ----
@backend.register("M")
def get_default_tensor_context():
    return default_context


@backend.register("M")
def set_default_tensor_context(**context):
    dtype = context.get("dtype", None)
    if isinstance(dtype, str):
        dtype = dtype_dict[dtype]
    default_context.update(dtype=dtype)


@backend.register("M")
def get_tensor_dtype(name):
    return dtype_dict[name]


@backend.register("M")
def get_tensor_device(name):
    raise NotImplementedError


@backend.register("M")
def is_tensor(tensor):
    return isinstance(tensor, np.ndarray)


@backend.register("M")
def context(tensor):
    return dict(dtype=tensor.dtype)


@backend.register("M")
def tensor(data, dtype=None):
    return np.array(data, dtype=dtype)


@backend.register("M")
def as_tensor(data, dtype=None):
    return np.asarray(data, dtype=dtype)


@backend.register("M")
def to_numpy(tensor):
    return np.asarray(tensor)


@backend.register("M")
def ndim(tensor):
    return tensor.ndim


@backend.register("M")
def diag(v, k=0, dtype=None) -> Tensor:
    return np.asarray(np.diag(v, k=k), dtype=dtype)


@backend.register("M")
def meshgrid(*tensors, indexing="ij", dtype=None):
    ret = np.meshgrid(*tensors, indexing=indexing)
    return [np.asarray(r, dtype=dtype) for r in ret]


@backend.register("M")
def index_update(tensor, indices, values):
    tensor[indices] = values
    return tensor


@backend.register("M")
def ravel_multi_range(low, high, shape, mode="raise", order="C"):
    multi_index = [range(low[j], high[j]) for j in range(len(shape))]
    multi_index = tuple(zip(*itertools.product(*multi_index)))
    return np.ravel_multi_index(multi_index, shape, mode=mode, order=order)


@backend.register("M")
def pad(
    tensor,
    width,
    mode="constant",
    stat_length=None,
    constant_values=None,
    end_values=0,
    reflect_type="even",
):
    if mode in ["constant"]:
        return np.pad(tensor, width, mode=mode, constant_values=constant_values)
    elif mode in ["maximum", "minimum", "mean", "median"]:
        return np.pad(tensor, width, mode=mode, stat_length=stat_length)
    elif mode in ["linear_ramp"]:
        return np.pad(tensor, width, mode=mode, end_values=end_values)
    elif mode in ["reflect"]:
        return np.pad(tensor, width, mode=mode, reflect_type=reflect_type)
    else:
        return np.pad(tensor, width, mode=mode)


@backend.register("M")
def nonzero(tensor, as_tuple=True):
    return np.nonzero(tensor)


@backend.register("M")
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


@backend.register("M")
def relu(x):
    return np.maximum(0, x)


@backend.register("M")
def gelu(x):
    coefficient = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(coefficient * (x + 0.044715 * x**3)))


@backend.register("M")
def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


@backend.register("M")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@backend.register("M")
def silu(x):
    return x / (1 + np.exp(-x))


@backend.register("M")
def convolve_along_axis(x, w, axis=0):
    func1d = lambda m: np.convolve(m, w, mode="valid")
    return np.apply_along_axis(func1d, axis, x)


@backend.register("M")
def moving_mean(x, n=2, axis=0):
    return convolve_along_axis(x, np.ones(n), axis=axis)


@backend.register("M")
def gkron(xs):
    k = len(xs)
    m = [np.size(_x, 0) for _x in xs]
    zs = []
    for i in range(k):
        zs.append(xs[i])
        for j in range(k):
            if i == j:
                continue
            elif i < j:
                zs[i] = np.kron(zs[i], np.ones((m[j], 1)))
            else:
                zs[i] = np.kron(np.ones((m[j], 1)), zs[i])
    return zs


# ----
for name in [
    # tensor creation
    "copy",
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "eye",
    "diagonal",
    "arange",
    "linspace",
    # tensor manipulation
    "size",
    "shape",
    "reshape",
    "ravel",
    "ravel_multi_index",
    "unravel_index",
    "transpose",
    "moveaxis",
    "swapaxes",
    "roll",
    "concatenate",
    "stack",
    "tile",
    "flip",
    "broadcast_to",
    # tensor sorting and searching
    "sort",
    "count_nonzero",
    "all",
    "any",
    "where",
    # math: part I
    "sign",
    "abs",
    "conj",
    "ceil",
    "floor",
    "round",
    "square",
    "sqrt",
    "exp",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "arcsinh",
    "arccosh",
    "arctanh",
    "clip",
    "maximum",
    "minimum",
    # math: part II
    "max",
    "min",
    "sum",
    "prod",
    "mean",
    "cumsum",
    "cumprod",
    # linear algebra
    "finfo",
    "logical_and",
    "logical_or",
    "add",
    "subtract",
    "multiply",
    "divide",
    "dot",
    "matmul",
    "tensordot",
    "einsum",
    "kron",
    "vectorize",
    # unit test
    "isclose",
]:
    backend.register_("M", name, getattr(np, name))

for name in ["norm", "qr", "svd", "eig", "solve", "lstsq"]:
    backend.register_("M", name, getattr(np.linalg, name))

for name in ["assert_allclose"]:
    backend.register_("M", name, getattr(np.testing, name))


# ----
@backend.register("M")
def rand(shape, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.rand(*shape)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def randn(shape, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.randn(*shape)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def gamma(shape, scale=1.0, size=None, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.gamma(shape=shape, scale=scale, size=size)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def choice(tensor, size=None, replace=True, p=None, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.choice(tensor, size=size, replace=replace, p=p)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def randint(low, high=None, size=None, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.randint(low, high=high, size=size)
    random_tensor = backend.get("tensor")(random_tensor, **context).astype(int)
    return random_tensor


@backend.register("M")
def permutation(x, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.permutation(x)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def uniform(low=0.0, high=1.0, size=None, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.uniform(low, high, size)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def normal(loc=0.0, scale=1.0, size=None, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.normal(loc, scale, size)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1.0
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


@backend.register("M")
def glorot_uniform(shape, scale=1.0, mode="fan_avg", seed=None, **context):
    rng = get_numpy_random_state(seed)
    fan_in, fan_out = _compute_fans(shape)
    if mode == "fan_in":
        scale /= max(1.0, fan_in)
    elif mode == "fan_out":
        scale /= max(1.0, fan_out)
    elif mode == "fan_avg":
        scale /= max(1.0, (fan_in + fan_out) / 2.0)
    limit = math.sqrt(3.0 * scale)
    random_tensor = rng.uniform(-limit, limit, shape)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def glorot_normal(
    shape, scale=1.0, mode="fan_avg", truncated=False, seed=None, **context
):
    rng = get_numpy_random_state(seed)
    fan_in, fan_out = _compute_fans(shape)
    if mode == "fan_in":
        scale /= max(1.0, fan_in)
    elif mode == "fan_out":
        scale /= max(1.0, fan_out)
    elif mode == "fan_avg":
        scale /= max(1.0, (fan_in + fan_out) / 2.0)
    stddev = math.sqrt(scale) / 0.87962566103423978 if truncated else math.sqrt(scale)
    random_tensor = rng.normal(0.0, stddev, shape)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def exponential(scale=1.0, size=None, seed=None, **context):
    rng = get_numpy_random_state(seed)
    random_tensor = rng.exponential(scale=scale, size=size)
    random_tensor = backend.get("tensor")(random_tensor, **context)
    return random_tensor


@backend.register("M")
def eps(dtype):
    return backend.get("finfo")(dtype).eps


# ----
def _preprocess_file(file, ext):
    if not file.endswith(ext):
        file += ext
    return file


@backend.register("M")
def save_to_mat(file, data):
    file = _preprocess_file(file, ".mat")
    savemat(file, data)


@backend.register("M")
def save_to_npy(file, data):
    file = _preprocess_file(file, ".npy")
    np.save(file, data)


@backend.register("M")
def load_from_mat(file):
    file = _preprocess_file(file, ".mat")
    return loadmat(file)


@backend.register("M")
def load_from_npy(file):
    file = _preprocess_file(file, ".mat")
    return np.load(file)


# TensorInterface.register_backend(backend.name, backend)
