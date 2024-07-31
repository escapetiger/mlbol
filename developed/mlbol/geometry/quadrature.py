import os
import warnings
import numpy as np
from typing import Tuple
from mlbol.dtensor import Tensor as Tensor
from mlbol.utils import current_func_name
from mlbol.geometry.transform import PolarToCartesian
from mlbol.geometry.transform import SphericalToCartesian
from mlbol.geometry.utils import out_as_tensor

_ASSET_DIR = os.path.dirname(__file__)
_ASSET_SPHERE = f"{_ASSET_DIR}/quad_sphere"

__all__ = [
    "quad_1d",
    "gauss_legendre_1d",
    "gauss_lobatto_1d",
    "gauss_chebyshev_1d",
    "quad_circle_2d",
    "gauss_circle_2d",
    "quad_sphere_3d",
    "sphere_lebedev_3d",
    "sphere_tdesign_3d",
]


# ========================================================
# Quadrature rules on [-1,1].
# ========================================================
def quad_1d(n: int, mode: str):
    r"""
    Generate quadrature coordinates and weights in :math:`[-1,1]`.

    By default, the integral is NOT normalized.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    quad : str
        One of the following:

        * "gausslegd" (Gauss-Legendre quadrature),
        * "gaussloba" (Gauss-Lobatto quadrature),
        * "gausscheb" (Gauss-Chebyshev quadrature),
        * "gausslagu" (Gauss-Laguerre quadrature).

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.
    """
    if mode == "gausslegd":
        return gauss_legendre_1d(n)
    if mode == "gaussloba":
        return gauss_lobatto_1d(n)
    if mode == "gausscheb":
        return gauss_chebyshev_1d(n)
    raise ValueError(f"{mode} quadrature is not available for {current_func_name()}.")


@out_as_tensor
def gauss_legendre_1d(n: int) -> Tuple[Tensor, Tensor]:
    r"""
    Gauss-Legendre quadrature.

    Exact for polynomials of degree :math:`2n-1` or less over the interval :math:`[-1,1]`
    with the weight function :math:`w(x)=1`.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


@out_as_tensor
def gauss_lobatto_1d(n: int) -> Tuple[Tensor, Tensor]:
    r"""
    Gauss-Lobatto quadrature.

    Exact for polynomials of degree :math:`2n-3` or less over the interval :math:`[-1,1]`
    with the weight function :math:`w(x)=1`.

    At least contains two endpoints, so n > 1.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.
    """
    # Truncation + 1
    n = n - 1
    N1 = n + 1

    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.pi * np.arange(n + 1) / n)

    # The Legendre Vandermonde Matrix
    P = np.zeros((N1, N1))

    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = np.ones(n + 1) * 2  # Initialize with a value different from x
    eps = np.finfo(float).eps  # Machine epsilon

    while np.max(np.abs(x - xold)) > eps:
        xold = x.copy()

        P[:, 0] = 1
        P[:, 1] = x

        for k in range(2, N1):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

        x = xold - (x * P[:, n] - P[:, n - 1]) / (N1 * P[:, n])

    w = 2 / (n * N1 * P[:, n] ** 2)

    return x, w


@out_as_tensor
def gauss_chebyshev_1d(n: int) -> Tuple[Tensor, Tensor]:
    r"""
    Gauss-Chebyshev quadrature.

    Exact for polynomials of degree :math:`2n-1` or less over the interval :math:`[-1,1]`
    with the weight function :math:`w(x)=1/\sqrt{1-x^2}`.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.
    """
    x, w = np.polynomial.chebyshev.chebgauss(n)
    return x, w


# ========================================================
# Quadrature rules on the 2D unit circle.
# ========================================================
def quad_circle_2d(n: int, mode: str) -> Tuple[Tensor, Tensor]:
    r"""
    Generate quadrature coordinates and weights on the unit circle.

    By default, the integral is normalized.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    quad : str
        One of the following:

        * "gausslegd" (Gauss-Legendre quadrature),
        * "gaussloba" (Gauss-Lobatto quadrature),
        * "gausscheb" (Gauss-Chebyshev quadrature),
        * "gausslagu" (Gauss-Laguerre quadrature).

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.

    References
    ----------
    * Daruis, Leyla, Pablo González-Vera, and Francisco Marcellán. "Gaussian quadrature formulae on the unit circle." Journal of computational and applied mathematics 140.1-2 (2002): 159-183.
    """
    if mode in ["gausslegd", "gaussloba", "gausscheb"]:
        return gauss_circle_2d(n, mode)
    raise ValueError(f"{mode} quadrature is not available for {current_func_name()}.")


@out_as_tensor
def gauss_circle_2d(n: int, mode: str) -> Tuple[Tensor, Tensor]:
    r"""
    Gauss-type quadrature on the unit circle.

    For practical computation, we rewrite the normalized integral as

    .. math::
        I(f) = 1/2\pi \int_{0}^{2\pi} f(\cos(\theta), \sin(\theta)) d\theta

    and directly approximate it by a 1D quadrature rule:

    .. math::
        I_n(f) = 1/2\pi \sum_{i} w_i f(\cos(\theta_i), \sin(\theta_i)).

    However, this treatment may destroy the good property of Gauss-type
    quadrature :math:`I_n(f)=I(f)` when :math:`f` is a "low degree" polynomial.

    In the future, we could study the Szego-type quadrature to get better
    performance.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    quad : str
        One of the following:

        * "gausslegd" (Gauss-Legendre quadrature),
        * "gaussloba" (Gauss-Lobatto quadrature),
        * "gausscheb" (Gauss-Chebyshev quadrature),
        * "gausslagu" (Gauss-Laguerre quadrature).
    polar_coord: bool
        Whether represented in polar coordinates.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.
    """

    if mode == "gausslegd":
        x, w = gauss_legendre_1d(n)
    elif mode == "gaussloba":
        x, w = gauss_lobatto_1d(n)
    elif mode == "gausscheb":
        x, w = gauss_chebyshev_1d(n)
    x = np.pi * (x + 1)
    x = np.hstack((np.ones_like(x)[:, np.newaxis], x[:, np.newaxis]))
    w = w / 2
    return PolarToCartesian.transform(x), w


# ========================================================
# Quadrature rules on the 3D unit sphere.
# ========================================================
def quad_sphere_3d(n: int, mode: str) -> Tuple[Tensor, Tensor]:
    r"""
    Generate quadrature coordinates and weights on the 3D unit sphere surface.

    By default, the integral is normalized.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    quad : str
        One of the following:

        * "lebedev" (Lebedev quadrature),
        * "sphhs" (t-design, Hardin & Sloane),
        * "sphws" (t-design, Womersley, symmetric),
        * "sphwns" (t-design, Womersley, non-symmetric).

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.

    References
    ----------
    * Beentjes, Casper HL. "Quadrature on a spherical surface." Working note from https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf (2015).
    """
    if mode == "lebedev":
        return sphere_lebedev_3d(n)
    if mode in ["sphhs", "sphws", "sphwns"]:
        return sphere_tdesign_3d(n, mode)
    raise ValueError(f"{mode} quadrature is not available for {current_func_name()}.")


@out_as_tensor
def sphere_lebedev_3d(n: int) -> Tuple[Tensor, Tensor]:
    r"""
    Lebedev quadrature rule on a 3D unit sphere surface.

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.
    """
    with open(f"{_ASSET_SPHERE}/lebedev/map.txt", "r") as file:
        for line in file:
            lhs, rhs = map(int, line.strip().split(" "))
            if lhs >= n or lhs == 5810:
                # data format: (azimuth, polar, weight)
                data = np.loadtxt(f"{_ASSET_SPHERE}/lebedev/lebedev_{rhs:03d}.txt")
                x = np.ones((data.shape[0], 3))
                x[:, 1] = data[:, 1] * np.pi / 180
                x[:, 2] = (data[:, 0] / 180 + 1) * np.pi
                w = data[:, 2]
                if lhs != n:
                    warnings.warn(f"{n} points requested, {lhs} points sampled.")
                return SphericalToCartesian.transform(x), w


@out_as_tensor
def sphere_tdesign_3d(n: int, mode: str = "sphhs") -> Tuple[Tensor, Tensor]:
    r"""
    Spherical :math:`t`-designs, exact for all spherical harmonics up to degree :math:`t`:

    Parameters
    ----------
    n : int
        Number of quadrature points.
    quad : str, optional
        One of the following:

        * "sphhs": set by Hardin & Sloane up to t=21,
        * "sphws": set by Womersley, symmetric grid (exact integration odd spherical harmonics), up to t=311,
        * "sphwns: set by Womersley, non-symmetric, up to t=180.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Quadrature coordinates and weights.
    """
    if mode == "sphws":
        dir = f"{_ASSET_SPHERE}/sphdesigns/WomersleySym"
        name = "ss"
    elif mode == "sphwns":
        dir = f"{_ASSET_SPHERE}/sphdesigns/WomersleyNonSym"
        name = "sf"
    elif mode == "sphhs":
        dir = f"{_ASSET_SPHERE}/sphdesigns/HardinSloane"
        name = "hs"

    with open(f"{dir}/map.txt", "r") as file:
        for line in file:
            lhs, rhs = map(int, line.strip().split(" "))
            if lhs >= n:
                x = np.loadtxt(f"{dir}/{name}{rhs:03d}.{lhs:05d}")
                w = np.ones(x.shape[0]) / x.shape[0]
                if lhs != n:
                    warnings.warn(f"{n} points requested, {lhs} points sampled.")
                return x, w
