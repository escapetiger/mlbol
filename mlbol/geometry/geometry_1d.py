from mlbol.dtensor import Tensor
from mlbol.dtensor import tensor
from mlbol.dtensor import as_tensor
from mlbol.dtensor import ravel
from mlbol.dtensor import all
from mlbol.dtensor import logical_and
from mlbol.dtensor import isclose
from mlbol.dtensor import reshape
from mlbol.dtensor import choice
from mlbol.dtensor import linspace
from mlbol.dtensor import full
from mlbol.dtensor import concatenate
from mlbol.dtensor import size
from mlbol.dtensor import ones
from mlbol.dtensor import zeros
from mlbol.dtensor import copy
from mlbol.geometry.base import Geometry
from mlbol.geometry.random import random_unit_hypercube
from mlbol.geometry.quadrature import quad_1d

__all__ = ["Interval", "TimeInterval"]


class Interval(Geometry):
    def __init__(self, a: float, b: float) -> None:
        super().__init__(1, (tensor([a]), tensor([b])), b - a, set([1, 2]))
        self.a: float = float(a)
        self.b: float = float(b)
        self.center: float = (a + b) / 2

    def inside(self, x: Tensor) -> Tensor:
        c1 = ravel(all(self.a <= x, axis=-1))
        c2 = ravel(all(x <= self.b, axis=-1))
        return logical_and(c1, c2)

    def on_boundary(self, x: Tensor) -> Tensor:
        c1 = ravel(isclose(x, [self.a]))
        c2 = ravel(isclose(x, [self.b]))
        return as_tensor(1 * c1 + 2 * c2, dtype=int)

    def boundary_normal(self, x: Tensor) -> Tensor:
        c1 = isclose(x, [self.a])
        c2 = isclose(x, [self.b])
        return -1.0 * c1 + 1.0 * c2

    def random_points(self, n: int, mode: str = "pseudo") -> Tensor:
        x = random_unit_hypercube(n, 1, mode=mode)
        return self.diam * x + self.a

    def random_boundary_points(self, n: int, mode: str = "pseudo") -> Tensor:
        if n == 2:
            return tensor([[self.a], [self.b]])
        return reshape(choice([self.a, self.b], n), (-1, 1))

    def uniform_points(self, n: int, mode: str = "full") -> Tensor:
        if mode == "full":
            x = linspace(self.a, self.b, num=n)
        if mode == "interior":
            x = linspace(self.a, self.b, num=n + 1, endpoint=False)[1:]
        return reshape(x, (-1, 1))

    def uniform_boundary_points(self, n: int) -> Tensor:
        if n == 1:
            return tensor([[self.a]])
        xl = full((n // 2, 1), self.a)
        xr = full((n - n // 2, 1), self.b)
        return concatenate((xl, xr), axis=0)

    def quadrature_points(
        self, n: int, mode: str = "pseudo", normalized: bool = True
    ) -> tuple[Tensor, Tensor]:
        if mode in ["pseudo", "lhs", "halton", "hammersley", "sobol"]:
            x = self.random_points(n, mode=mode)
            w = ones((size(x, 0),)) / size(x, 0)
        else:
            x, w = quad_1d(n, mode)
            x = reshape(x, (-1, 1))
            w = reshape(w, (-1, 1))
            x = self.center + self.diam / 2 * x
            w /= 2
        return (x, w) if normalized else (x, w * self.diam)

    def periodic_points(self, x: Tensor, component: int = 0) -> Tensor:
        tmp = copy(x)
        mask_a = isclose(x, self.a)
        mask_b = isclose(x, self.b)
        mask_c = ~(mask_a & mask_b)
        tmp[mask_a] = self.b
        tmp[mask_b] = self.a
        tmp[mask_c] = self.a + (tmp[mask_c] - self.a) * (self.b - self.a)
        return tmp


class TimeInterval(Interval):
    def __init__(self, b: float) -> None:
        super().__init__(0.0, b)

    def on_boundary(self, x: Tensor) -> Tensor:
        c = ravel(isclose(x, [0.0]))
        return as_tensor(c, dtype=int)

    def random_boundary_points(self, n: int, mode: str = "pseudo") -> Tensor:
        return zeros((n, 1))

    def uniform_boundary_points(self, n: int):
        return zeros((n, 1))
