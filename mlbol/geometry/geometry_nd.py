import math, itertools, warnings
from mlbol.dtensor import Tensor
from mlbol.dtensor import tensor
from mlbol.dtensor import prod
from mlbol.dtensor import norm
from mlbol.dtensor import logical_and
from mlbol.dtensor import all
from mlbol.dtensor import any
from mlbol.dtensor import zeros
from mlbol.dtensor import isclose
from mlbol.dtensor import count_nonzero
from mlbol.dtensor import randint
from mlbol.dtensor import arange
from mlbol.dtensor import ceil
from mlbol.dtensor import linspace
from mlbol.dtensor import size
from mlbol.dtensor import copy
from mlbol.dtensor import ravel
from mlbol.dtensor import as_tensor
from mlbol.dtensor import reshape
from mlbol.dtensor import ones
from mlbol.dtensor import round
from mlbol.dtensor import pi
from mlbol.geometry.base import Geometry
from mlbol.geometry.random import random_unit_hypercube
from mlbol.geometry.random import random_unit_hyperball
from mlbol.geometry.random import random_unit_hypersphere
from mlbol.geometry.quadrature import quad_1d
from mlbol.geometry.quadrature import quad_circle_2d
from mlbol.geometry.quadrature import quad_sphere_3d

__all__ = ["HyperRectangle", "HyperBall", "HyperSphere"]


class HyperRectangle(Geometry):
    def __init__(self, xmin: Tensor, xmax: Tensor) -> None:
        assert len(xmin) == len(xmax), "Dimensions of xmin and xmax do not match."

        self.xmin: Tensor = tensor(xmin)
        self.xmax: Tensor = tensor(xmax)

        assert all(self.xmin < self.xmax), "Constraint xmin < xmax do not hold."

        self.xlen: Tensor = self.xmax - self.xmin
        self.center: Tensor = (self.xmax + self.xmin) / 2
        self.volume: float = float(prod(self.xlen))

        super().__init__(
            len(xmin),
            (self.xmin, self.xmax),
            norm(self.xlen),
            bdrs=set(range(1, 2 ** len(xmin) + 1)),
        )

    def inside(self, x: Tensor) -> Tensor:
        return logical_and(all(x >= self.xmin, axis=-1), all(x <= self.xmax, axis=-1))

    def on_boundary(self, x: Tensor) -> Tensor:
        c = zeros(x.shape[:-1], dtype=int)
        for d in range(self.ndim):
            for k, xb in enumerate([self.xmin[d], self.xmax[d]]):
                mask = isclose(x[..., d], xb)
                c[mask] = d * 2 + k + 1
        return c

    def boundary_normal(self, x: Tensor) -> Tensor:
        _n = -1.0 * isclose(x, self.xmin) + 1.0 * isclose(x, self.xmax)
        # For vertices, the normal is averaged for all directions
        idx = count_nonzero(_n, axis=-1) > 1
        if any(idx):
            print(
                f"Warning: {self.__class__.__name__} boundary_normal called on vertices. "
            )
            l = norm(_n[idx], axis=-1, keepdims=True)
            _n[idx] /= l
        return _n

    def random_points(self, n: int, mode: str = "pseudo") -> Tensor:
        x = random_unit_hypercube(n, self.ndim, mode=mode)
        return self.xlen * x + self.xmin

    def random_boundary_points(
        self, n: int, mode: str = "pseudo", component: tuple[int, int] = None
    ) -> Tensor:
        """One can set component=(dim, 0 or 1) to focus on a subset of boundary."""
        x = random_unit_hypercube(n, self.ndim, mode=mode)
        if component is None:
            # randomly pick a dimension
            rand_dim = randint(0, high=self.ndim, size=n)
            # replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
            x[arange(n), rand_dim] = round(x[arange(n), rand_dim])
        else:
            x[arange(n), component[0]] = component[1]
        return self.xlen * x + self.xmin

    def uniform_points(self, n: int, mode: str = "full") -> Tensor:
        dx = (self.volume / n) ** (1 / self.ndim)
        xd = []
        for d in range(self.ndim):
            nd = int(ceil(self.xlen[d] / dx))
            if mode == "full":
                xd.append(linspace(self.xmin[d], self.xmax[d], num=nd))
            if mode == "interior":
                xd.append(
                    linspace(
                        self.xmin[d],
                        self.xmax[d],
                        num=nd + 1,
                        endpoint=False,
                    )[1:]
                )
        x = tensor(list(itertools.product(*xd)))
        if n != size(x, 0):
            print(f"Warning: {n} points requested, but {size(x, 0)} points sampled.")
        return x

    def quadrature_points(
        self, n: int, mode: str = "gausslegd", normalized: bool = True
    ) -> tuple[Tensor, Tensor]:
        dx = (self.volume / n) ** (1 / self.ndim)
        xd, wd = [], []
        for d in range(self.ndim):
            nd = int(ceil(self.xlen[d] / dx))
            xdr, wdr = quad_1d(nd, mode)
            xd.append(self.xlen[d] / 2 * xdr + self.center[d])
            wd.append(self.xlen[d] / 2 * wdr)
        x = tensor(list(itertools.product(*xd)))
        w = prod(tensor(list(itertools.product(*wd))), axis=-1, keepdims=True)
        if n != len(x):
            print(f"Warning: {n} points requested, but {len(x)} points sampled.")
        return (x, w) if normalized else (x, w / self.volume)

    def periodic_points(self, x: Tensor, component: int) -> Tensor:
        y = copy(x)
        _on_xmin = isclose(y[:, component], self.xmin[component])
        _on_xmax = isclose(y[:, component], self.xmax[component])
        y[:, component][_on_xmin] = self.xmax[component]
        y[:, component][_on_xmax] = self.xmin[component]
        return y


class HyperBall(Geometry):

    def __init__(self, center: Tensor, radius: float) -> None:
        self.center: Tensor = tensor(center)
        self.radius: float = float(radius)
        self.volume: float
        super().__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )
        n = self.ndim
        if n == 1:
            self.volume = 2 * radius
        elif n == 2:
            self.volume = pi * radius**2
        elif n == 3:
            self.volume = 4 / 3 * pi * radius**3
        else:
            self.volume = pi ** (n / 2) / math.gamma(n / 2 + 1) * radius ** (n)

    def inside(self, x: Tensor) -> Tensor:
        return norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x: Tensor) -> Tensor:
        c = ravel(isclose(norm(x - self.center, axis=-1), self.radius))
        return as_tensor(c, dtype=int)

    def boundary_normal(self, x: Tensor) -> Tensor:
        _n = x - self.center
        l = reshape(norm(_n, axis=-1), (-1, 1))
        _n = _n / l * isclose(l, self.radius)
        return _n

    def random_points(self, n: int, mode: str = "pseudo") -> Tensor:
        x = random_unit_hyperball(n, self.ndim, mode=mode)
        x = self.radius * x + self.center
        return x

    def random_boundary_points(self, n: int, mode: str = "pseudo") -> Tensor:
        x = random_unit_hypersphere(n, self.ndim, mode=mode)
        x = self.radius * x + self.center
        return x

    def quadrature_points(
        self,
        n: int,
        mode: str = "pseudo",
        normalized: bool = True,
    ) -> tuple[Tensor, Tensor]:
        x = self.random_points(n, mode=mode)
        w = ones((size(x, 0),)) / size(x, 0)
        w = reshape(w, (-1, 1))
        if mode not in ["pseudo", "lhs", "halton", "hammersley", "sobol"]:
            warnings.warn(f"{mode} points requested, but Monte-Carlo points sampled.")
        return (x, w) if normalized else (x, w * self.volume)


class HyperSphere(Geometry):
    def __init__(self, center: Tensor, radius: float) -> None:
        self.center: Tensor = tensor(center)
        self.radius: float = float(radius)
        self.real_ndim: int
        self.area: float
        super().__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )
        n = self.ndim
        self.real_ndim = n - 1
        if n == 1:
            self.area = 2.0
        elif n == 2:
            self.area = 2 * pi * radius
        elif n == 3:
            self.area = 4 * pi * radius**2
        else:
            self.area = 2 * pi ** (n / 2) / math.gamma(n / 2) * radius ** (n - 1)

    def inside(self, x: Tensor) -> Tensor:
        return isclose(norm(x - self.center, axis=-1), self.radius)

    def random_points(self, n: int, mode: str = "pseudo") -> Tensor:
        x = random_unit_hypersphere(n, self.ndim, mode=mode)
        x = self.radius * x + self.center
        return x

    def quadrature_points(
        self,
        n: int,
        mode: str = "pseudo",
        normalized: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if mode in ["pseudo", "lhs", "halton", "hammersley", "sobol"]:
            x = self.random_points(n, mode=mode)
            w = ones((size(x, 0),)) / size(x, 0)
            if mode != "montecarlo":
                warnings.warn(
                    f"{mode} points requested, but Monte-Carlo points sampled."
                )
        else:
            if self.ndim == 2:
                x, w = quad_circle_2d(n, mode)
                x = self.radius * x + self.center
            elif self.ndim == 3:
                x, w = quad_sphere_3d(n, mode)
                x = self.radius * x + self.center
        w = reshape(w, (-1, 1))
        return (x, w) if normalized else (x, w * self.area)
