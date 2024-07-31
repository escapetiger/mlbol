import warnings
from typing import Tuple
from typing import List
from mlbol.dtensor import Tensor
from mlbol.dtensor import norm

__all__ = ["Geometry"]


class Geometry:
    r"""An abstract base class for geometries."""

    def __init__(
        self, ndim: int, bbox: Tuple[Tensor], diam: float, bdrs: set = set()
    ) -> None:
        self.ndim: int = ndim
        self.bbox: Tuple[Tensor] = bbox
        self.diam: float = min(diam, norm(bbox[1] - bbox[0]))
        self.bdrs: set = bdrs

    def __repr__(self) -> str:
        return type(self).__name__

    def inside(self, x: Tensor) -> Tensor:
        r"""Check if `x` is inside the geometry (including the boundary).

        Parameters
        ----------
        x : Tensor
            Point coordinates.

        Returns
        -------
        Tensor
            True if `x` is inside the geometry, otherwise False.
        """
        raise NotImplementedError

    def on_boundary(self, x: Tensor) -> Tensor:
        r"""Check if `x` is on the geometry boundary.

        Return an integer indicating at which boundary a
        point located. If it is `0`, the point is not on the
        boundary of this geometry.

        Parameters
        ----------
        x : Tensor
            Point coordinates.

        Returns
        -------
        Tensor
            Integer tensor indicating the relative position of points
            in terms of boundary.
        """
        raise NotImplementedError

    def boundary_normal(self, x: Tensor) -> Tensor:
        r"""Draw the unit normal at boundary points `x`.

        Parameters
        ----------
        x : Tensor
            Point coordinates.

        Returns
        -------
        Tensor
            True if `x` is on the geometry boundary, otherwise False.
        """
        raise NotImplementedError

    def random_points(self, n: int, **generator) -> Tensor:
        r"""Draw random points within the geometry.

        Parameters
        ----------
        n : int
            Number of random points.
        mode : {'pseudo', 'lhs', 'halton', 'hammersley', 'sobol'}, optional
            Algorithm for random sequences. Available choices are:
                - 'pseudo': pseudorandom sequence
                - 'lhs': Latin hypercube sampling
                - 'halton': Halton sequence
                - 'sobol': Sobol sequence
        Returns
        -------
        Tensor
            Point coordinates.
        """
        raise NotImplementedError

    def random_boundary_points(self, n: int, **generator) -> Tensor:
        r"""Draw random points on the geometry boundary.

        Parameters
        ----------
        n : int
            Number of random points.
        mode : {'pseudo', 'lhs', 'halton', 'hammersley', 'sobol'}, optional
            Algorithm for random sequences. Available choices are:
                - 'pseudo': pseudorandom sequence
                - 'lhs': Latin hypercube sampling
                - 'halton': Halton sequence
                - 'sobol': Sobol sequence
        Returns
        -------
        Tensor
            Point coordinates.
        """
        raise NotImplementedError

    def uniform_points(self, n: int, **generator) -> Tensor:
        r"""Draw uniform points within the geometry.

        Parameters
        ----------
        n : int
            Number of uniform points.
        mode : {'full', 'interior'}, optional
            Algorithm for uniform partition. Available choices are:
                - 'full': including boundary points
                - 'interior': no boundary points

        Returns
        -------
        Tensor
            Point coordinates.
        """
        warnings.warn(f"{self}.uniform_points is undefined. Use random_points instead.")
        return self.random_points(n)

    def uniform_boundary_points(self, n: int) -> Tensor:
        r"""Draw uniform points on the geometry boundary.

        Parameters
        ----------
        n : int
            Number of uniform points.

        Returns
        -------
        Tensor
            Point coordinates.
        """
        warnings.warn(
            f"{self}.uniform_boundary_points is undefined. Use random_boundary_points instead."
        )
        return self.random_boundary_points(n)

    def quadrature_points(self, n: int, **generator) -> Tuple[Tensor, Tensor]:
        r"""Draw quadrature points within the geometry.

        Parameters
        ----------
        n : int
            Number of quadrature points.
        mode: str
            Quadrature rule. The default rule is the Monte-Carlo rule with
            'pseudo' generator. Besides, all available random generators
            are listed as follows:

            * 'pseudo' (default): pseudorandom sequence
            * 'lhs': Latin hypercube sampling
            * 'halton': Halton sequence
            * 'sobol': Sobol sequence

            For Interval and HyperRectangle objects, it can be chosen to be:

            * 'gausslegd': Gauss-Legendre rule
            * 'gaussloba': Gauss-Lobatto rule
            * 'gausscheb': Gauss-Chebyshev rule

            For HyperSphere objects, it can be chosen to be:

            * 1D:

                ** 'gausslegd': Gauss-Legendre rule
                ** 'gaussloba': Gauss-Lobatto rule
                ** 'gausscheb': Gauss-Chebyshev rule

            * 2D:

                ** 'gausslegd': Gauss-Legendre rule
                ** 'gaussloba': Gauss-Lobatto rule
                ** 'gausscheb': Gauss-Chebyshev rule

            * 3D:

                ** 'lebedev': Lebedev rule
                ** 'sphhs': sphere t-design rule by Hardin & Sloane
                ** 'sphws': symmetric sphere t-design rule by Womersley
                ** 'sphwns': non-symmetric sphere t-design rule by Womersley
        normalized: bool, optional
            Determine whether normalize the integral. Default is True.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Quadrature coordinates and weights, both stored in a 2D Tensor.
        """
        raise NotImplementedError

    def periodic_points(self, x: Tensor, component: int) -> Tensor:
        r"""Compute the periodic image of `x` for periodic boundary condition.

        Parameters
        ----------
        x : Tensor
            Point coordinates.
        component : int
            Periodic component.

        Returns
        -------
        Tensor
            Perioidic images.
        """
        raise NotImplementedError

    def limits(self, extend: float = 0.0) -> List[List[float]]:
        """Draw an extended bounding box.

        Parameters
        ----------
        extend : float, optional
            Extension parameter. The default value is 0.

        Returns
        -------
        List[List[float]]
            Extended bounding box.
        """
        lim = []
        for d in range(self.ndim):
            _min = self.bbox[0][d] - max(1, abs(self.bbox[0][d])) * extend
            _max = self.bbox[1][d] + max(1, abs(self.bbox[1][d])) * extend
            lim.append([_min, _max])
        return lim
