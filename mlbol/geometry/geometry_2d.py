import warnings
from typing import Sequence
from mlbol.dtensor import Tensor
from mlbol.dtensor import sum
from mlbol.dtensor import ceil
from mlbol.dtensor import stack
from mlbol.dtensor import linspace
from mlbol.dtensor import full
from mlbol.dtensor import concatenate
from mlbol.dtensor import size
from mlbol.dtensor import isclose
from mlbol.dtensor import prod
from mlbol.dtensor import ones
from mlbol.dtensor import reshape
from mlbol.dtensor import cos
from mlbol.dtensor import sin
from mlbol.dtensor import pi
from mlbol.geometry.geometry_nd import HyperRectangle
from mlbol.geometry.geometry_nd import HyperSphere
from mlbol.geometry.random import random_circle
from mlbol.geometry.quadrature import quad_circle_2d

__all__ = ["Rectangle", "Circle"]


class Rectangle(HyperRectangle):
    def __init__(self, xmin: Tensor, xmax: Tensor) -> None:
        assert len(xmin) == 2, "a Rectangle object must be 2D."
        super().__init__(xmin, xmax)
        self.perimeter: float = 2 * float(sum(self.xmax - self.xmin))
        self.area: float = self.volume

    def uniform_boundary_points(self, n: int) -> Tensor:
        nx, ny = ceil(n / self.perimeter * (self.xmax - self.xmin))
        nx, ny = int(nx), int(ny)
        # avoid repeatly generating corner points
        xb = stack(
            (
                linspace(self.xmin[0], self.xmax[0], num=nx, endpoint=False),
                full((nx,), self.xmin[1]),
            ),
            axis=-1,
        )
        xr = stack(
            (
                full((ny,), self.xmax[0]),
                linspace(self.xmin[1], self.xmax[1], num=ny, endpoint=False),
            ),
            axis=-1,
        )
        xt = stack(
            (
                linspace(self.xmin[0], self.xmax[0], num=nx + 1)[1:],
                full((nx,), self.xmax[1]),
            ),
            axis=-1,
        )
        xl = stack(
            (
                full((ny,), self.xmin[0]),
                linspace(self.xmin[1], self.xmax[1], num=ny + 1)[1:],
            ),
            axis=-1,
        )
        x = concatenate((xb, xr, xt, xl), axis=0)
        if n != size(x, 0):
            warnings.warn(f"{n} points requested, but {size(x, 0)} points sampled.")
        return x

    @staticmethod
    def is_valid(vertices: Sequence[Tensor]):
        """Check if the geometry is a Rectangle."""
        return (
            len(vertices) == 4
            and isclose(prod(vertices[1] - vertices[0]), 0)
            and isclose(prod(vertices[2] - vertices[1]), 0)
            and isclose(prod(vertices[3] - vertices[2]), 0)
            and isclose(prod(vertices[0] - vertices[3]), 0)
        )


class Circle(HyperSphere):
    """
    By default, 2D Circle is of center=(0,0) and radius=1.
    """

    def __init__(self, center: Tensor = [0, 0], radius: float = 1) -> None:
        assert len(center) == 2, "A Sphere object must be 2D."
        super().__init__(center, radius)

    def random_points(self, n: int, mode: str = "pseudo") -> Tensor:
        x = random_circle(n, mode=mode)
        return self.radius * x + self.center

    def quadrature_points(
        self, n: int, mode: str = "pseudo", normalized: bool = True
    ) -> tuple[Tensor, Tensor]:
        if mode in ["pseudo", "lhs", "halton", "hammersley", "sobol"]:
            x = self.random_points(n, mode=mode)
            w = ones((size(x, 0),)) / size(x, 0)
        else:
            x, w = quad_circle_2d(n, mode)
            x = self.radius * x + self.center
        w = reshape(w, (-1, 1))
        return (x, w) if normalized else (x, w * self.area)

    def uniform_points(self, n: int, mode: str = "full") -> Tensor:
        t = linspace(0, 2 * pi, n)
        x = stack([cos(t), sin(t)], axis=-1)
        return self.radius * x + self.center
