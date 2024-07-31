import warnings, itertools
from mlbol.dtensor import Tensor
from mlbol.dtensor import sum
from mlbol.dtensor import roll
from mlbol.dtensor import ceil
from mlbol.dtensor import linspace
from mlbol.dtensor import tensor
from mlbol.dtensor import concatenate
from mlbol.dtensor import full
from mlbol.dtensor import reshape
from mlbol.dtensor import size
from mlbol.dtensor import ones
from mlbol.dtensor import stack
from mlbol.dtensor import sin
from mlbol.dtensor import cos
from mlbol.dtensor import pi
from mlbol.geometry.geometry_nd import HyperRectangle
from mlbol.geometry.geometry_nd import HyperSphere
from mlbol.geometry.random import random_sphere
from mlbol.geometry.quadrature import quad_sphere_3d

__all__ = ["Cuboid", "Sphere"]


class Cuboid(HyperRectangle):
    r"""3D Cuboid."""

    def __init__(self, xmin: Tensor, xmax: Tensor) -> None:
        assert len(xmin) == 3, "A Cuboid object must be 3D."
        super().__init__(xmin, xmax)
        dx = self.xmax - self.xmin
        self.area: float = 2 * float(sum(dx * roll(dx, 2)))

    # def random_boundary_points(self, n: int, random: str = "pseudo") -> Tensor:
    #     pts = []
    #     density = n / self.area
    #     rect = Rectangle(self.xmin[:-1], self.xmax[:-1])
    #     for z in [self.xmin[-1], self.xmax[-1]]:
    #         u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
    #         pts.append(np.hstack((u, np.full((len(u), 1), z))))
    #     rect = Rectangle(self.xmin[::2], self.xmax[::2])
    #     for y in [self.xmin[1], self.xmax[1]]:
    #         u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
    #         pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), y), u[:, 1:])))
    #     rect = Rectangle(self.xmin[1:], self.xmax[1:])
    #     for x in [self.xmin[0], self.xmax[0]]:
    #         u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
    #         pts.append(np.hstack((np.full((len(u), 1), x), u)))
    #     pts = np.vstack(pts)
    #     if len(pts) > n:
    #         return pts[np.random.choice(len(pts), size=n, replace=False)]
    #     return pts

    def uniform_boundary_points(self, n: int) -> Tensor:
        h = (self.area / n) ** 0.5
        nx, ny, nz = ceil((self.xmax - self.xmin) / h) + 1
        nx, ny, nz = int(nx), int(ny), int(nz)
        x = linspace(self.xmin[0], self.xmax[0], num=nx)
        y = linspace(self.xmin[1], self.xmax[1], num=ny)
        z = linspace(self.xmin[2], self.xmax[2], num=nz)

        # avoid repeatly generating corner points
        pts = []
        for v in [self.xmin[-1], self.xmax[-1]]:
            u = tensor(list(itertools.product(x, y)))
            pts.append(concatenate((u, full((len(u), 1), v)), axis=-1))
        if nz > 2:
            for v in [self.xmin[1], self.xmax[1]]:
                u = tensor(list(itertools.product(x, z[1:-1])))
                pts.append(
                    concatenate((u[:, 0:1], full((len(u), 1), v), u[:, 1:]), axis=-1)
                )
        if ny > 2 and nz > 2:
            for v in [self.xmin[0], self.xmax[0]]:
                u = tensor(list(itertools.product(y[1:-1], z[1:-1])))
                pts.append(concatenate((full((len(u), 1), v), u), axis=-1))
        pts = concatenate(pts, axis=0)
        if n != size(pts, 0):
            warnings.warn(
                f"Warning: {n} points requested, but {size(pts, 0)} points sampled."
            )
        return pts


class Sphere(HyperSphere):
    r"""By default, 3D Sphere is of center=(0,0,0) and radius=1."""

    def __init__(self, center: Tensor = [0, 0, 0], radius: float = 1) -> None:
        assert len(center) == 3, "A Sphere object must be 3D."
        super().__init__(center, radius)

    def random_points(
        self,
        n: int,
        mode: str = "pseudo",
        symmetric: bool = False,
    ) -> Tensor:
        x = random_sphere(n, mode=mode, symmetric=symmetric)
        return self.radius * x + self.center

    def quadrature_points(
        self,
        n: int,
        mode: str = "montecarlo",
        normalized: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if mode in ["pseudo", "lhs", "halton", "hammersley", "sobol"]:
            x = self.random_points(n, mode=mode)
            w = ones((size(x, 0),)) / size(x, 0)
        else:
            x, w = quad_sphere_3d(n, mode)
            x = self.radius * x + self.center
        w = reshape(w, (-1, 1))
        return (x, w) if normalized else (x, w * self.area)

    def uniform_points(self, n: int, mode: str = "full") -> Tensor:
        t = linspace(0, pi, n)
        s = linspace(0, 2 * pi, n)
        x = stack([sin(t) * cos(s), sin(t) * cos(s), cos(t)], axis=-1)
        return self.radius * x + self.center
