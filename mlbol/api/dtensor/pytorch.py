import builtins
import itertools
import math
import numpy as np

try:
    import torch
    from torch.nn import functional as F
except ImportError as error:
    message = (
        "Fail to import PyTorch.\n"
        "To use DeepMLT with the PyTorch backend, "
        "you must first install PyTorch!"
    )
    raise ImportError(message) from error

from typing import Any, Dict
from mlbol.utils import GroupedRegistry
# from mlbol.src.dtensor.interface import TensorInterface
from mlbol.api.dtensor.utils import get_numpy_random_state, savemat, loadmat

backend = GroupedRegistry("pytorch", group=("A", "M"))

default_context = {
    "dtype": torch.float32,
    "device": torch.device("cpu"),
    "requires_grad": False,
}
dtype_dict = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


# ---- Attributes ----
@backend.register("A")
class Tensor(torch.Tensor):
    @property
    def context(self) -> Dict[str, Any]:
        return {
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
        }


for name in [
    # types
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
    backend.register_("A", name, getattr(torch, name))


# ---- Methods ----
@backend.register("M")
def get_default_tensor_context():
    return default_context


@backend.register("M")
def set_default_tensor_context(**context):
    dtype = context.get("dtype", None)
    device = context.get("device", None)
    requires_grad = context.get("requires_grad", False)
    if isinstance(dtype, str):
        dtype = dtype_dict[dtype]
    if isinstance(device, str):
        device = torch.device(device)
    default_context.update(dtype=dtype, device=device, requires_grad=requires_grad)
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)


@backend.register("M")
def get_tensor_dtype(name):
    return dtype_dict[name]


@backend.register("M")
def get_tensor_device(name):
    return torch.device(name)


@backend.register("M")
def context(tensor):
    return dict(dtype=tensor.dtype, device=tensor.device, requires_grad=tensor.requires_grad)


@backend.register("M")
def tensor(data, dtype=torch.float32, device="cpu", requires_grad=False):
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


@backend.register("M")
def as_tensor(data, dtype=torch.float32, device="cpu", requires_grad=False):
    return torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(
        requires_grad
    )


@backend.register("M")
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.asarray(tensor)


@backend.register("M")
def copy(tensor):
    return torch.clone(tensor)


@backend.register("M")
def eye(N, M=None, **context):
    return torch.eye(N, **context)


@backend.register("M")
def diag(v, k=0, **context):
    return torch.as_tensor(torch.diag(v, diagonal=k), **context)


@backend.register("M")
def diagonal(tensor, offset=0, axis1=0, axis2=1) -> Tensor:
    return torch.diagonal(tensor, offset=offset, dim1=axis1, dim2=axis2)


@backend.register("M")
def arange(start=0, stop=None, step=None, **context):
    if step is None:
        step = 1
    if stop is None:
        return torch.arange(0, start, step, **context)
    else:
        return torch.arange(start, stop, step, **context)


@backend.register("M")
def linspace(start, stop, num, endpoint=True, **context):
    if endpoint:
        res = torch.linspace(start, stop, num, **context)
    else:
        res = torch.linspace(start, stop, num + 1, **context)[:-1]
    return res


@backend.register("M")
def meshgrid(*tensors, indexing="ij", **context):
    ret = torch.meshgrid(*tensors, indexing=indexing)
    return [torch.as_tensor(r, **context) for r in ret]


@backend.register("M")
def size(tensor, axis=None):
    return tensor.size(dim=axis) if axis is not None else torch.numel(tensor)


@backend.register("M")
def shape(tensor):
    return tensor.shape


@backend.register("M")
def ndim(tensor):
    return tensor.dim()


@backend.register("M")
def index_update(tensor, indices, values):
    tensor[indices] = values
    return tensor


@backend.register("M")
def ravel_multi_index(multi_index, shape, mode="raise", order="C"):
    # IMPL ONE
    if mode not in ["raise", "wrap", "clip"]:
        raise ValueError(f"valid mode is {'raise', 'wrap', 'clip'} but get '{mode}'")
    if order not in ["C", "F"]:
        raise ValueError(f"valid order is {'C', 'F'} but get '{order}'")
    multi_index = torch.tensor(multi_index, dtype=torch.long).T
    shape = torch.tensor(shape, dtype=torch.long)
    if mode == "raise":
        if torch.any(torch.any(multi_index >= shape, dim=0)):
            raise ValueError(
                f"shape is {shape} but multi-indices are out of range:\n {multi_index}"
            )
    if mode == "wrap":
        multi_index = multi_index % shape[None, :]
    if mode == "clip":
        multi_index = torch.clip(multi_index, torch.zeros_like(shape), shape - 1)
    prods = torch.ones_like(shape)
    if order == "C":
        prods[:-1] = torch.flip(torch.cumprod(torch.flip(shape[1:], (0,)), 0), (0,))
    if order == "F":
        prods[1:] = torch.cumprod(shape[:-1], 0)
    ret = torch.matmul(multi_index, prods)
    return ret
    # IMPL TWO
    # multi_index = to_numpy(multi_index)
    # ret = np.ravle_multi_index(multi_index, shape, mode=mode, order=order)
    # return torch.tensor(ret)


@backend.register("M")
def ravel_multi_range(low, high, shape, mode="raise", order="C"):
    multi_index = [range(low[j], high[j]) for j in range(len(shape))]
    multi_index = tuple(zip(*itertools.product(*multi_index)))
    return ravel_multi_index(multi_index, shape, mode=mode, order=order)


@backend.register("M")
def unravel_index(indices, shape, order="C") -> tuple[Tensor]:
    # IMPL ONE
    if order not in ["C", "F"]:
        raise ValueError(f"valid order is {'C', 'F'} but get '{order}'")
    indices = torch.tensor(indices, dtype=torch.long)
    shape = torch.tensor(shape, dtype=torch.long)
    prods = torch.ones_like(shape)
    if order == "C":
        prods[:-1] = torch.flip(torch.cumprod(torch.flip(shape[1:], (0,)), 0), (0,))
    if order == "F":
        prods[1:] = torch.cumprod(shape[:-1], 0)
    indices = torch.reshape(indices, (-1, 1))
    prods = torch.reshape(prods, (1, -1))
    shape = torch.reshape(shape, (1, -1))
    multi_index = torch.floor_divide(indices, prods) % shape
    multi_index = torch.hsplit(multi_index, multi_index.shape[1])
    return tuple(x.squeeze() for x in multi_index)
    # IMPL TWO
    # indices = to_numpy(indices)
    # ret = np.unravel_index(indices, shape, order=order)
    # return torch.tensor(ret)


@backend.register("M")
def transpose(tensor, axes=None):
    if axes is None:
        axes = list(range(tensor.ndim))[::-1]
    return tensor.permute(*axes)


@backend.register("M")
def concatenate(tensors, axis=0):
    return torch.concat(tensors, dim=axis)


@backend.register("M")
def stack(tensors, axis=0):
    return torch.stack(tensors, dim=axis)


@backend.register("M")
def flip(tensors, axis=None):
    return torch.flip(tensors, dims=(axis,) if not isinstance(axis, tuple) else axis)


@backend.register("M")
def broadcast_to(tensor, shape):
    return tensor.expand(*shape)


@backend.register("M")
def roll(tensors, shift, axis=None):
    if axis is None:
        axis = ()
    else:
        axis = (axis,) if not isinstance(axis, tuple) else axis
    return torch.roll(tensors, shift, dims=axis)


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
    if backend.get("is_tensor"):
        original_context = backend.get("context")(tensor)
    else:
        tensor = backend.get("as_tensor")(tensor)
        original_context = {}
    tensor = backend.get("to_numpy")(tensor)
    if mode in ["constant"]:
        tensor = np.pad(tensor, width, mode=mode, constant_values=constant_values)
    elif mode in ["maximum", "minimum", "mean", "median"]:
        tensor = np.pad(tensor, width, mode=mode, stat_length=stat_length)
    elif mode in ["linear_ramp"]:
        tensor = np.pad(tensor, width, mode=mode, end_values=end_values)
    elif mode in ["reflect"]:
        tensor = np.pad(tensor, width, mode=mode, reflect_type=reflect_type)
    else:
        tensor = np.pad(tensor, width, mode=mode)
    tensor = backend.get("tensor")(tensor, **original_context)
    return tensor
    # assert len(width) == 2 * builtins.min(
    #     int(tensor.ndim), 3
    # ), "pad size should be double of min(tensor.ndim, 3)"
    # tensor = as_tensor(tensor)
    # if mode == "constant":
    #     return F.pad(tensor, width, mode=mode, value=value)
    # elif mode == "wrap":
    #     # In circular mode, tensor dimension should be of n + 2.
    #     if tensor.ndim <= 3:
    #         return F.pad(tensor[None, None, :], width, mode="circular")[0, 0]
    #     elif tensor.ndim == 4:
    #         return F.pad(tensor[None, :], width, mode="circular")[0]
    # elif mode == "edge":
    #     # In replicate mode, tensor dimension should be of n + 1.
    #     if tensor.ndim <= 4:
    #         return F.pad(tensor[None, :], width, mode="replicate")[0]
    # elif mode == "reflect":
    #     # In replicate mode, tensor dimension should be of n + 1.
    #     if tensor.ndim <= 4:
    #         return F.pad(tensor[None, :], width, mode="reflect")[0]
    # return F.pad(tensor, width, mode=mode)


@backend.register("M")
def sort(tensor, axis=-1):
    return torch.sort(tensor, dim=axis)


@backend.register("M")
def count_nonzero(tensor, axis=None):
    return torch.count_nonzero(tensor, dim=axis)


@backend.register("M")
def all(tensor, axis=None):
    if axis is None:
        return torch.all(tensor)
    return torch.all(tensor, dim=axis)


@backend.register("M")
def any(tensor, axis=None):
    if axis is None:
        return torch.any(tensor)
    return torch.any(tensor, dim=axis)


@backend.register("M")
def max(x, axis=None, keepdims=False):
    if axis is None:
        return torch.max(x)
    return torch.max(x, dim=axis, keepdim=keepdims)


@backend.register("M")
def min(x, axis=None, keepdims=False):
    if axis is None:
        return torch.min(x)
    return torch.min(x, dim=axis, keepdim=keepdims)


@backend.register("M")
def sum(x, axis=None, keepdims=False):
    if axis is None:
        return torch.sum(x)
    return torch.sum(x, dim=axis, keepdim=keepdims)


@backend.register("M")
def prod(x, axis=None, keepdims=False):
    if axis is None:
        return torch.prod(x)
    return torch.prod(x, dim=axis, keepdim=keepdims)


@backend.register("M")
def cumsum(x, axis=None):
    if axis is None:
        return torch.cumsum(x.flatten())
    return torch.cumsum(x, dim=axis)


@backend.register("M")
def cumprod(x, axis=None):
    if axis is None:
        return torch.cumprod(x.flatten())
    return torch.cumprod(x, dim=axis)


@backend.register("M")
def mean(x, axis=None, keepdims=False):
    if axis is None:
        return torch.mean(x)
    return torch.mean(x, dim=axis, keepdim=keepdims)


@backend.register("M")
def convolve_along_axis(x, w, axis=0):
    # To ensure consistency with numpy, we flips weight tensor before calling torch.conv.
    weight = w.to(x)
    if x.ndim == 1:
        input = x.reshape(1, 1, -1)
        weight = torch.flip(weight, dims=(0,)).reshape(1, 1, -1)
        result = F.conv1d(input, weight, stride=1, padding=0)
        result = result.squeeze()
        return result

    input = x.unfold(axis, len(weight), 1)
    result = torch.matmul(input, torch.flip(weight, dims=(0,)))
    return result


@backend.register("M")
def moving_mean(x, n=2, axis=0):
    if x.ndim == 1:
        input = x.reshape(1, 1, -1)
        weight = torch.ones(1, 1, n).to(x)
        result = F.conv1d(input, weight, stride=1, padding=0)
        result = result.squeeze() / n
        return result

    input = x.unfold(axis, n, 1)
    weight = torch.ones(n).to(x)
    result = torch.matmul(input, weight) / n
    return result


@backend.register("M")
def norm(tensor, ord=None, axis=None, keepdims=False):
    # pytorch does not accept `None` for any keyword arguments. additionally,
    # pytorch doesn't seems to support keyword arguments in the first place
    kwds = {}
    if axis is not None:
        kwds["dim"] = axis
    if ord and ord != "inf":
        kwds["p"] = ord
    kwds["keepdim"] = keepdims

    if ord == "inf":
        res = torch.max(torch.abs(tensor), **kwds)
        if axis is not None:
            return res[0]  # ignore indices output
        return res
    return torch.norm(tensor, **kwds)


@backend.register("M")
def dot(a, b):
    if a.ndim > 2 and b.ndim > 2:
        return torch.tensordot(a, b, dims=([-1], [-2]))
    if not a.ndim or not b.ndim:
        return a * b
    return torch.matmul(a, b)


@backend.register("M")
def tensordot(a, b, axes=2):
    return torch.tensordot(a, b, dims=axes)


@backend.register("M")
def gkron(xs):
    k = len(xs)
    m = [_x.shape[0] for _x in xs]
    zs = []
    for i in range(k):
        zs.append(xs[i])
        for j in range(k):
            if i == j:
                continue
            elif i < j:
                zs[i] = torch.kron(zs[i], torch.ones(m[j], 1).to(zs[i]))
            else:
                zs[i] = torch.kron(torch.ones(m[j], 1).to(zs[i]), zs[i])
    return zs


# ----
for name in [
    # tensor creation
    "is_tensor",
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    # tensor manipulation
    "reshape",
    "ravel",
    "moveaxis",
    "swapaxes",
    "tile",
    # tensor sorting and searching
    "where",
    "nonzero",
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
    # linear algebra
    "logical_and",
    "logical_or",
    "add",
    "subtract",
    "multiply",
    "divide",
    "matmul",
    "einsum",
    "kron",
]:
    backend.register_("M", name, getattr(torch, name))

for name in ["elu", "relu", "gelu", "selu", "sigmoid", "silu"]:
    backend.register_("M", name, getattr(torch.nn.functional, name))

for name in ["qr", "svd", "eig", "solve", "lstsq"]:
    backend.register_("M", name, getattr(torch.linalg, name))


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
    random_tensor = backend.get("tensor")(random_tensor, **context).int()
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
        scale /= builtins.max(1.0, fan_in)
    elif mode == "fan_out":
        scale /= builtins.max(1.0, fan_out)
    elif mode == "fan_avg":
        scale /= builtins.max(1.0, (fan_in + fan_out) / 2.0)
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
    return finfo(dtype).eps


@backend.register("M")
def finfo(dtype):
    return np.finfo(
        backend.get("to_numpy")(backend.get("tensor")([], dtype=dtype)).dtype
    )


@backend.register("M")
def vectorize(func):
    return torch.vmap(func)


# ----
class _Jacobian:
    def __init__(self, ys: torch.Tensor, xs: torch.Tensor) -> None:
        self.J = {}
        self.ys = ys
        self.xs = xs
        self.ny = ys.shape[1]
        self.nx = xs.shape[1]

    def __call__(self, i: int = 0, j: int | None = None):
        if not 0 <= i < self.ny:
            raise ValueError("i={} is not valid.".format(i))
        if j is not None and not 0 <= j < self.nx:
            raise ValueError("j={} is not valid.".format(j))
        # Compute J[i]
        if i not in self.J:
            # TODO: retain_graph=True has memory leak?
            y = self.ys[:, i : i + 1] if self.ny > 1 else self.ys
            self.J[i] = torch.autograd.grad(
                y,
                self.xs,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True,
            )[0]

        return self.J[i] if j is None or self.nx == 1 else self.J[i][:, j : j + 1]


class _LazyJacobian:
    Js = {}

    @classmethod
    def compute(
        cls, ys: torch.Tensor, xs: torch.Tensor, i: int = 0, j: int | None = None
    ):
        key = (ys, xs)
        if key not in cls.Js:
            cls.Js[key] = _Jacobian(ys, xs)
        return cls.Js[key](i, j)

    @classmethod
    def clear(cls) -> None:
        cls.Js = {}


@backend.register("M")
def jacobian(
    ys: torch.Tensor, xs: torch.Tensor, i: int = 0, j: int = None, lazy: bool = True
):
    if lazy:
        return _LazyJacobian.compute(ys, xs, i, j)
    return _Jacobian(ys, xs)(i, j)


class _Hessian:
    def __init__(
        self,
        ys: torch.Tensor,
        xs: torch.Tensor,
        i: int = None,
    ) -> None:
        ny = ys.shape[1]

        if ny > 1:
            if i is None:
                raise ValueError("The component of y is missing.")
            if i >= ny:
                raise ValueError(
                    "The component of y={} cannot be larger than the dimension={}.".format(
                        i, ny
                    )
                )
        else:
            if i is not None:
                raise ValueError("Do not use component for 1D y.")
            i = 0

        grad_y_i = jacobian(ys, xs, i=i, j=None, lazy=True)
        self.H = _Jacobian(grad_y_i, xs)

    def __call__(self, i: int = 0, j: int = 0) -> torch.Tensor:
        return self.H(i, j)


class _LazyHessian:
    Hs = {}

    @classmethod
    def compute(
        cls,
        ys: torch.Tensor,
        xs: torch.Tensor,
        i: int | None = None,
        j: int = 0,
        k: int = 0,
    ) -> torch.Tensor:
        key = (ys, xs, i)
        if key not in cls.Hs:
            cls.Hs[key] = _Hessian(ys, xs, i)
        return cls.Hs[key](j, k)

    @classmethod
    def clear(cls) -> None:
        cls.Hs = {}


@backend.register("M")
def hessian(
    ys: torch.Tensor,
    xs: torch.Tensor,
    i: int | None = None,
    j: int = 0,
    k: int = 0,
    lazy: bool = True,
) -> torch.Tensor:
    if lazy:
        return _LazyHessian.compute(ys, xs, i, j, k)
    return _Hessian(ys, xs, i)(j, k)


# ----
def _preprocess_file(file, ext):
    if not file.endswith(ext):
        file += ext
    return file


def _preprocess_data(data):
    if isinstance(data, torch.Tensor):
        return to_numpy(data)
    elif isinstance(data, dict):
        return {key: _preprocess_data(val) for key, val in data.items()}
    return data


def _postprocess_data(data):
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        return tensor(data)
    elif isinstance(data, dict):
        return {key: _postprocess_data(val) for key, val in data.items()}
    elif isinstance(data, list):
        return [_postprocess_data(val) for val in data]
    return data


@backend.register("M")
def save_to_mat(file, data):
    file = _preprocess_file(file, ".mat")
    data = _preprocess_data(data)
    savemat(file, data)


@backend.register("M")
def save_to_npy(file, data):
    file = _preprocess_file(file, ".npy")
    data = _preprocess_data(data)
    np.save(file, data, allow_pickle=True)


@backend.register("M")
def load_from_mat(file):
    file = _preprocess_file(file, ".mat")
    data = loadmat(file)
    return _postprocess_data(data)


@backend.register("M")
def load_from_npy(file):
    file = _preprocess_file(file, ".npy")
    data = np.load(file, allow_pickle=True)
    return _postprocess_data(data)


# ----
@backend.register("M")
def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@backend.register("M")
def assert_allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, err_msg=None):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    torch.testing.assert_close(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=err_msg
    )


