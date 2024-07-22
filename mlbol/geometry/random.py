import numpy as np
from scipy import stats as st
from mlbol.dtensor import Tensor as Tensor
from mlbol.geometry.transform import SphericalToCartesian
from mlbol.geometry.utils import out_as_tensor

__all__ = [
    "random_unit_hypercube",
    "random_unit_hyperball",
    "random_unit_hypersphere",
    "random_circle",
    "random_sphere",
]


def _pseudorandom(n: int, ndim: int) -> np.ndarray:
    r"""Generate pseudo-random numbers.

    A pseudorandom sequence of numbers is one that appears to be statistically random,
    despite having been produced by a completely deterministic and repeatable process.

    If random seed is set, then the rng based code always returns the same random
    number, which may not be what we expect:.
        rng = np.random.default_rng(config.random_seed)
        return rng.random(size=(n, ndim), dtype=kbe.np_dtype)

    Parameters
    ----------
    n : int
        Number of samples.
    ndim : int
        Sample space dimension.

    Returns
    -------
    Tensor
        The sample coordinates.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pseudorandomness
    """

    return np.random.random(size=(n, ndim))


def _quasirandom(n: int, ndim: int, mode: str) -> np.ndarray:
    r"""Generate quasi-random numbers.

    A low-discrepancy sequence, which is also called quasi-random sequence,
    is a sequence with the property that for all values of :math:`N`,
    its subsequence :math:`x_1, ..., x_N` has a low discrepancy.

    Parameters
    ----------
    n : int
        Number of samples.
    ndim : int
        Sample space dimension.
    mode : str
        One of the following:

        - "lhs" (Latin hypercube sampling),
        - "halton" (Halton sequence),
        - "sobol" (Sobol sequence).

    Returns
    -------
    Tensor
        The sample coordinates.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Low-discrepancy_sequence
    .. [2] https://en.wikipedia.org/wiki/Latin_hypercube_sampling
    .. [3] https://en.wikipedia.org/wiki/Halton_sequence
    .. [4] https://mathworld.wolfram.com/HammersleyPointSet.html
    .. [5] https://en.wikipedia.org/wiki/Sobol_sequence
    """

    if mode == "lhs":
        # mode = skopt.sampler.Lhs()
        mode = st.qmc.LatinHypercube(ndim)
    elif mode == "halton":
        mode = st.qmc.Halton(ndim)
    elif mode == "sobol":
        mode = st.qmc.Sobol(ndim)
    return np.asarray(mode.random(n))


@out_as_tensor
def random_unit_hypercube(n: int, ndim: int, mode: str = "pseudo") -> Tensor:
    r"""Generate pseudorandom or quasirandom samples in the unit hypercube, i.e.,
    :math:`[0,1]^{ndim}`.

    Parameters
    ----------
    n : int
        Number of samples.
    ndim : int
        Sample space dimension.
    random : str, optional
        One of the following:

        - "pseudo" (pseudorandom, by default),
        - "lhs" (Latin hypercube sampling),
        - "halton" (Halton sequence),
        - "sobol" (Sobol sequence).

    Returns
    -------
    Tensor
        The sample coordinates.
    """
    if mode == "pseudo":
        return _pseudorandom(n, ndim)
    if mode in ["lhs", "halton", "hammersley", "sobol"]:
        return _quasirandom(n, ndim, mode=mode)
    raise ValueError("f{sampler} sampling is not available.")


@out_as_tensor
def random_unit_hyperball(n: int, ndim: int, mode: str = "pseudo") -> Tensor:
    """Generate pseudorandom or quasirandom samples in the unit hyperball.

    Parameters
    ----------
    n : int
        Number of samples.
    ndim : int
        Sample space dimension.
    random : str, optional
        One of the following:

        - "pseudo" (pseudorandom, by default),
        - "lhs" (Latin hypercube sampling),
        - "halton" (Halton sequence),
        - "sobol" (Sobol sequence).

    Returns
    -------
    Tensor
        The sample coordinates.
    """
    if mode == "pseudo":
        # https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        u = np.random.rand(n, 1)
        x = np.random.normal(size=(n, ndim))
    else:
        rng = random_unit_hypercube(n, ndim + 1, mode=mode)
        u, x = rng[:, 0:1], rng[:, 1:]  # Error if X = [0, 0, ...]
        x = st.norm.ppf(x)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return u ** (1 / ndim) * x


@out_as_tensor
def random_unit_hypersphere(n: int, ndim: int, mode: str = "pseudo") -> Tensor:
    """Generate pseudorandom or quasirandom samples on the unit hypersphere.

    Parameters
    ----------
    n : int
        Number of samples.
    ndim : int
        Sample space dimension.
    random : str, optional
        One of the following:

        - "pseudo" (pseudorandom, by default),
        - "lhs" (Latin hypercube sampling),
        - "halton" (Halton sequence),
        - "hammersley" (Hammersley sequence),
        - "sobol" (Sobol sequence).

    Returns
    -------
    Tensor
        The sample coordinates.
    """
    if mode == "pseudo":
        # http://mathworld.wolfram.com/HyperspherePointPicking.html
        x = np.random.normal(size=(n, ndim))
    else:
        # Error for [0, 0, ...] or [0.5, 0.5, ...]
        u = random_unit_hypercube(n, ndim, mode=mode)
        x = st.norm.ppf(u)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    return x


@out_as_tensor
def random_circle(n: int, mode: str = "pseudo") -> Tensor:
    x = random_unit_hypersphere(n, 2, mode=mode)
    return x


@out_as_tensor
def random_sphere(n: int, mode: str = "pseudo", symmetric: bool = False) -> Tensor:
    if not symmetric:
        x = random_unit_hypersphere(n, 3, mode=mode)
    else:
        if n % 8 != 0:
            n = (n // 8 + 1) * 8

        m = n // 8
        r = random_unit_hypercube(m, 2, mode=mode)
        x_sub = np.ones((m, 3))
        x_sub[:, 1] = 1 / 2 * np.arccos(2 * r[:, 0] - 1)
        x_sub[:, 2] = r[:, 1] * np.pi / 2
        x_sub = SphericalToCartesian.transform(x_sub)

        x = np.empty((n, 3))
        x[:m, :] = x_sub * np.array([1, 1, 1])[np.newaxis, :]
        x[m : 2 * m, :] = x_sub * np.array([-1, 1, 1])[np.newaxis, :]
        x[2 * m : 3 * m, :] = x_sub * np.array([1, -1, 1])[np.newaxis, :]
        x[3 * m : 4 * m, :] = x_sub * np.array([1, 1, -1])[np.newaxis, :]
        x[4 * m : 5 * m, :] = x_sub * np.array([-1, -1, 1])[np.newaxis, :]
        x[5 * m : 6 * m, :] = x_sub * np.array([-1, 1, -1])[np.newaxis, :]
        x[6 * m : 7 * m, :] = x_sub * np.array([1, -1, -1])[np.newaxis, :]
        x[7 * m : 8 * m, :] = x_sub * np.array([-1, -1, -1])[np.newaxis, :]
    return x
