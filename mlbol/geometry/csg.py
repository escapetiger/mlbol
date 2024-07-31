from mlbol.dtensor import Tensor
from mlbol.dtensor import minimum
from mlbol.dtensor import maximum
from mlbol.dtensor import reshape
from mlbol.dtensor import copy
from mlbol.dtensor import empty
from mlbol.dtensor import concatenate
from mlbol.dtensor import permutation
from mlbol.dtensor import logical_and
from mlbol.dtensor import rand
from mlbol.geometry.base import Geometry

# note: CSG geometry has an empty set `bdrs`!

__all__ = [
    "CSGUnion",
    "CSGDifference",
    "CSGIntersection",
    "csg_union",
    "csg_difference",
    "csg_intersection",
]


class CSGUnion(Geometry):
    """Construct a geometry by CSG Union."""

    geom1: Geometry
    geom2: Geometry

    def __init__(self, geom1: Geometry, geom2: Geometry) -> None:
        if geom1.ndim != geom2.ndim:
            raise ValueError(f"{geom1} | {geom2} failed (dimensions do not match).")
        super().__init__(
            geom1.ndim,
            (
                minimum(geom1.bbox[0], geom2.bbox[0]),
                maximum(geom1.bbox[1], geom2.bbox[1]),
            ),
            geom1.diam + geom2.diam,
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x: Tensor) -> Tensor:
        return self.geom1.inside(x) | self.geom2.inside(x)

    def on_boundary(self, x: Tensor) -> Tensor:
        cond_a = self.geom1.on_boundary(x) & (~self.geom2.inside(x))
        cond_b = self.geom2.on_boundary(x) & (~self.geom1.inside(x))
        return cond_a | cond_b

    def boundary_normal(self, x: Tensor) -> Tensor:
        a = self.geom1.on_boundary(x) & (~self.geom2.inside(x))
        b = self.geom2.on_boundary(x) & (~self.geom1.inside(x))
        a = reshape(a, (-1, 1)) * self.geom1.boundary_normal(x)
        b = reshape(b, (-1, 1)) * self.geom2.boundary_normal(x)
        return a + b

    def random_points(self, n: int, random: str = "pseudo") -> Tensor:
        """
        To generate random points in a CSGUnion object,
        we use a while-loop to sample points repeatly in its `bbox`,
        collecting the samples that locate inside `geom1` or `geom2`.
        """
        x = empty(shape=(n, self.ndim))
        i = 0
        while i < n:
            tmp = rand(n, self.ndim) * (self.bbox[1] - self.bbox[0]) + self.bbox[0]
            tmp = tmp[self.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n: int, random: str = "pseudo") -> Tensor:
        """
        To generate random boundary points in a CSGUnion object,
        we use a while-loop to sample points repeatly on the boundaries
        of `geom1` and `geom2`, collecting the samples that locate
        on the boundary of CSGUnion geometry.
        """
        x = empty(shape=(n, self.ndim))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                ~self.geom1.inside(geom2_boundary_points)
            ]

            tmp = concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_points(self, x: Tensor, component: int) -> Tensor:
        """
        To generate periodic points in a CSGUnion object,
        we call `geom1.periodic_points` to move points that locate
        on the boundary of `geom1` but outside of `geom2`, and then
        call `geom2.periodic_points` to move points that locate on
        the boundary of `geom2` but outside of `geom1`.
        """
        x = copy(x)
        on_boundary_geom1 = logical_and(
            self.geom1.on_boundary(x), ~self.geom2.inside(x)
        )
        x[on_boundary_geom1] = self.geom1.periodic_points(x, component)[
            on_boundary_geom1
        ]
        on_boundary_geom2 = logical_and(
            self.geom2.on_boundary(x), ~self.geom1.inside(x)
        )
        x[on_boundary_geom2] = self.geom2.periodic_points(x, component)[
            on_boundary_geom2
        ]
        return x


class CSGDifference(Geometry):
    """Construct a geometry by CSG Difference."""

    geom1: Geometry
    geom2: Geometry

    def __init__(self, geom1: Geometry, geom2: Geometry) -> None:
        if geom1.ndim != geom2.ndim:
            raise ValueError(
                "{} - {} failed (dimensions do not match).".format(geom1, geom2)
            )
        super().__init__(geom1.ndim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x: Tensor) -> Tensor:
        return self.geom1.inside(x) & (~self.geom2.inside(x))

    def on_boundary(self, x: Tensor) -> Tensor:
        a = self.geom1.on_boundary(x) & (~self.geom2.inside(x))
        b = self.geom1.inside(x) & self.geom2.on_boundary(x)
        return a | b

    def boundary_normal(self, x: Tensor) -> Tensor:
        a = self.geom1.on_boundary(x) & (~self.geom2.inside(x))
        b = self.geom1.inside(x) & self.geom2.on_boundary(x)
        a = reshape(a, (-1, 1)) * self.geom1.boundary_normal(x)
        b = reshape(b, (-1, 1)) * self.geom2.boundary_normal(x)
        return a - b

    def random_points(self, n: int, random: str = "pseudo") -> Tensor:
        """
        To generate random points in a CSGDifference Gobject,
        we use a while-loop to sample points repeatly in its `bbox`,
        collecting the samples that locate inside of `geom1` but outside of `geom2`.
        """
        x = empty(shape=(n, self.ndim))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[~self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n: int, random: str = "pseudo") -> Tensor:
        """
        To generate random boundary points in a CSGDifference object,
        we use a while-loop to sample points repeatly on the boundaries
        of `geom1` and `geom2`, collecting the samples that locate
        on the boundary of CSGDifference geometry.
        """
        x = empty(shape=(n, self.ndim))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_points(self, x: Tensor, component: int) -> Tensor:
        """
        To generate periodic points in a CSGDifference object,
        we call `geom1.periodic_points` to move points that locate
        on the boundary of `geom1` but outside of `geom2`.
        """
        x = copy(x)
        on_boundary_geom1 = logical_and(
            self.geom1.on_boundary(x), ~self.geom2.inside(x)
        )
        x[on_boundary_geom1] = self.geom1.periodic_points(x, component)[
            on_boundary_geom1
        ]
        return x


class CSGIntersection(Geometry):
    """Construct a geomtry by CSG Intersection."""

    geom1: Geometry
    geom2: Geometry

    def __init__(self, geom1: Geometry, geom2: Geometry) -> None:
        if geom1.ndim != geom2.ndim:
            raise ValueError(
                "{} & {} failed (dimensions do not match).".format(geom1, geom2)
            )
        super().__init__(
            geom1.ndim,
            (
                maximum(geom1.bbox[0], geom2.bbox[0]),
                minimum(geom1.bbox[1], geom2.bbox[1]),
            ),
            min(geom1.diam, geom2.diam),
        )
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x: Tensor) -> Tensor:
        return self.geom1.inside(x) & self.geom2.inside(x)

    def on_boundary(self, x: Tensor) -> Tensor:
        a = self.geom1.on_boundary(x) & self.geom2.inside(x)
        b = self.geom1.inside(x) & self.geom2.on_boundary(x)
        return a | b

    def boundary_normal(self, x: Tensor) -> Tensor:
        a = self.geom1.on_boundary(x) & self.geom2.inside(x)
        b = self.geom1.inside(x) & self.geom2.on_boundary(x)
        a = reshape(a, (-1, 1)) * self.geom1.boundary_normal(x)
        b = reshape(b, (-1, 1)) * self.geom2.boundary_normal(x)
        return a + b

    def random_points(self, n: int, random: str = "pseudo") -> Tensor:
        """
        To generate random points in a CSGIntersection object,
        we use a while-loop to sample points repeatly in its `bbox`,
        collecting the samples that locate inside `geom1` and `geom2`.
        """
        x = empty(shape=(n, self.ndim))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n: int, random: str = "pseudo") -> Tensor:
        """
        To generate random boundary points in a CSGIntersection object,
        we use a while-loop to sample points repeatly on the boundaries
        of `geom1` and `geom2`, collecting the samples that locate
        on the boundary of CSGIntersection geometry.
        """
        x = empty(shape=(n, self.ndim))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[
                self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def periodic_points(self, x: Tensor, component: int) -> Tensor:
        """
        To generate periodic points in a CSGIntersection object,
        we call `geom1.periodic_points` to move points that locate
        on the boundary of `geom1` and inside of `geom2`, and call
        `geom2.periodic_points` to move points that locate on the
        boundary of `geom2` and inside of `geom1`.
        """
        x = copy(x)
        on_boundary_geom1 = logical_and(self.geom1.on_boundary(x), self.geom2.inside(x))
        x[on_boundary_geom1] = self.geom1.periodic_points(x, component)[
            on_boundary_geom1
        ]
        on_boundary_geom2 = logical_and(self.geom2.on_boundary(x), self.geom1.inside(x))
        x[on_boundary_geom2] = self.geom2.periodic_points(x, component)[
            on_boundary_geom2
        ]
        return x


def csg_union(geom1: Geometry, geom2: Geometry) -> CSGUnion:
    """Returns the union of `geom1` and `geom2`."""
    return CSGUnion(geom1, geom2)


def csg_difference(geom1: Geometry, geom2: Geometry) -> CSGDifference:
    """Returns the difference of `geom1` and `geom2`."""
    return CSGDifference(geom1, geom2)


def csg_intersection(geom1: Geometry, geom2: Geometry) -> CSGIntersection:
    """Returns the intersection of `geom1` and `geom2`."""
    return CSGIntersection(geom1, geom2)
