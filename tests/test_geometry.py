import unittest
import itertools
import math
import mlbol as dl
from mlbol import dtensor as T
from mlbol import geometry as G

dl.set_backend("pytorch")
# dl.set_backend("numpy")

from tests.utils import assert_equal, assert_almost_equal


class TestInterval(unittest.TestCase):
    def test_init(self):
        geom = G.Interval(0.0, 1.0)
        assert_equal(geom.a, 0.0)
        assert_equal(geom.b, 1.0)
        assert_equal(str(geom), "Interval")
        assert_equal(geom.bbox, (T.tensor([0]), T.tensor([1])))
        assert_equal(geom.diam, 1.0)

    def test_random_points(self):
        geom = G.Interval(0.0, 1.0)
        x = geom.random_points(10)
        assert_equal(T.shape(x), (10, 1))
        assert_equal(T.all(geom.inside(x)), True)

    def test_random_boundary_points(self):
        geom = G.Interval(0.0, 1.0)
        x = geom.random_boundary_points(10)
        assert_equal(T.shape(x), (10, 1))
        assert_equal(T.all(geom.on_boundary(x)), True)

    def test_uniform_points(self):
        geom = G.Interval(0.0, 1.0)
        x = geom.uniform_points(10)
        assert_equal(T.shape(x), (10, 1))
        assert_almost_equal(x, T.reshape(T.linspace(0, 1, 10), (-1, 1)))

    def test_uniform_boundary_points(self):
        geom = G.Interval(0.0, 1.0)
        x = geom.uniform_boundary_points(10)
        x_true = T.reshape(T.tensor([0] * 5 + [1] * 5), (-1, 1))
        assert_equal(T.shape(x), (10, 1))
        assert_almost_equal(x, x_true)

    def test_quadrature_points(self):
        geom = G.Interval(0.0, 1.0)
        x, w = geom.quadrature_points(3, "gausslegd")
        x_true = T.reshape(T.tensor([-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]), (-1, 1))
        w_true = T.reshape(T.tensor([5 / 9, 8 / 9, 5 / 9]), (-1, 1))
        x_true = x_true / 2 + 1 / 2
        w_true = w_true / 2
        assert_almost_equal(x, x_true)
        assert_almost_equal(w, w_true)


class TestRectangle(unittest.TestCase):
    def test_init(self):
        geom = G.Rectangle([-1, 0], [1, 1])
        assert_equal(geom.xlen, [2, 1])
        assert_equal(geom.center, [0, 1 / 2])
        assert_equal(geom.volume, 2)
        assert_equal(geom.area, 2)
        assert_equal(str(geom), "Rectangle")
        assert_equal(geom.bbox[0], [-1, 0])
        assert_equal(geom.bbox[1], [1, 1])
        assert_equal(geom.diam, math.sqrt(5))

    def test_random_points(self):
        geom = G.Rectangle([-1, 0], [1, 1])
        x = geom.random_points(100)
        assert_equal(T.shape(x), (100, 2))
        assert_equal(T.all(geom.inside(x)), True)

    def test_random_boundary_points(self):
        geom = G.Rectangle([-1, 0], [1, 1])
        x = geom.random_boundary_points(100)
        assert_equal(T.shape(x), (100, 2))
        assert_equal(T.all(geom.on_boundary(x)), True)

    def test_uniform_points(self):
        geom = G.Rectangle([0, 0], [1, 1])
        x = geom.uniform_points(100)
        x_true = T.linspace(0, 1, 10)
        x_true = T.tensor(list(itertools.product(x_true, repeat=2)))
        assert_almost_equal(x, x_true)

    def test_uniform_boundary_points(self):
        geom = G.Rectangle([0, 0], [1, 1])
        x = geom.uniform_boundary_points(100)
        assert_equal(T.shape(x), (100, 2))
        assert_equal(T.all(geom.on_boundary(x)), True)

    def test_quadrature_points(self):
        geom = G.Rectangle([0, 0], [1, 1])
        x, w = geom.quadrature_points(9, "gausslegd")
        x_true = [-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]
        x_true = T.tensor(list(itertools.product(x_true, repeat=2)))
        x_true = x_true / 2 + 1 / 2
        w_true = T.tensor([5 / 9, 8 / 9, 5 / 9]) / 2
        w_true = T.prod(
            T.tensor(list(itertools.product(w_true, repeat=2))), axis=1, keepdims=True
        )
        assert_almost_equal(x, x_true)
        assert_almost_equal(w, w_true)


class TestCircle(unittest.TestCase):
    def test_init(self):
        geom = G.Circle([0, 0], 1)
        assert_equal(str(geom), "Circle")
        assert_equal(geom.center, [0, 0])
        assert_equal(geom.radius, 1)
        assert_equal(geom.bbox[0], [-1, -1])
        assert_equal(geom.bbox[1], [1, 1])
        assert_almost_equal(geom.diam, 2)
        assert_almost_equal(geom.area, 2 * T.pi)

    def test_random_points(self):
        geom = G.Circle([0, 0], 1)
        x = geom.random_points(100)
        assert_equal(T.shape(x), (100, 2))
        assert_equal(T.all(geom.inside(x)), True)

    def test_quadrature_points(self):
        geom = G.Circle([0, 0], 1)
        x, w = geom.quadrature_points(30, "gausslegd")
        f = T.ones((T.size(x, 0), 1))
        assert_equal(T.shape(x), (30, 2))
        assert_equal(T.shape(w), (30, 1))
        assert_equal(T.all(geom.inside(x)), True)
        assert_almost_equal(T.transpose(w) @ f, T.tensor([[1]]))


class TestCuboid(unittest.TestCase):
    def test_init(self):
        geom = G.Cuboid([-1, 0, 1], [1, 1, 2])
        assert_equal(geom.xlen, [2, 1, 1])
        assert_equal(geom.center, [0, 1 / 2, 3 / 2])
        assert_equal(geom.volume, 2)
        assert_equal(geom.area, 10)
        assert_equal(str(geom), "Cuboid")
        assert_equal(geom.bbox[0], [-1, 0, 1])
        assert_equal(geom.bbox[1], [1, 1, 2])
        assert_equal(geom.diam, math.sqrt(6))

    def test_random_points(self):
        geom = G.Cuboid([-1, 0, 1], [1, 1, 2])
        x = geom.random_points(100)
        assert_equal(T.shape(x), (100, 3))
        assert_equal(T.all(geom.inside(x)), True)

    def test_random_boundary_points(self):
        geom = G.Cuboid([-1, 0, 1], [1, 1, 2])
        x = geom.random_boundary_points(100)
        assert_equal(T.shape(x), (100, 3))
        assert_equal(T.all(geom.on_boundary(x)), True)

    def test_uniform_points(self):
        geom = G.Cuboid([0, 0, 0], [1, 1, 1])
        x = geom.uniform_points(27)
        x_true = T.linspace(0, 1, 3)
        x_true = T.tensor(list(itertools.product(x_true, repeat=3)))
        assert_almost_equal(x, x_true)

    def test_uniform_boundary_points(self):
        geom = G.Cuboid([0, 0, 0], [1, 1, 1])
        x = geom.uniform_boundary_points(6 * 64)
        assert_equal(T.shape(x), (386, 3))
        assert_equal(T.all(geom.on_boundary(x)), True)

    def test_quadrature_points(self):
        geom = G.Cuboid([0, 0, 0], [1, 1, 1])
        x, w = geom.quadrature_points(9, "gausslegd")
        x_true = [-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]
        x_true = T.tensor(list(itertools.product(x_true, repeat=3)))
        x_true = x_true / 2 + 1 / 2
        w_true = T.tensor([5 / 9, 8 / 9, 5 / 9]) / 2
        w_true = T.prod(
            T.tensor(list(itertools.product(w_true, repeat=3))), axis=1, keepdims=True
        )
        assert_almost_equal(x, x_true)
        assert_almost_equal(w, w_true)


class TestHyperRectangle(unittest.TestCase):
    def test_init(self):
        geom = G.HyperRectangle([-1, 0], [1, 1])
        assert_equal(geom.xlen, [2, 1])
        assert_equal(geom.center, [0, 1 / 2])
        assert_equal(geom.volume, 2)
        assert_equal(str(geom), "HyperRectangle")
        assert_equal(geom.bbox[0], [-1, 0])
        assert_equal(geom.bbox[1], [1, 1])
        assert_equal(geom.diam, math.sqrt(5))

    def test_random_points(self):
        geom = G.HyperRectangle([-1, 0], [1, 1])
        x = geom.random_points(100)
        assert_equal(T.shape(x), (100, 2))
        assert_equal(T.all(geom.inside(x)), True)

    def test_random_boundary_points(self):
        geom = G.HyperRectangle([-1, 0], [1, 1])
        x = geom.random_boundary_points(100)
        assert_equal(T.shape(x), (100, 2))
        assert_equal(T.all(geom.on_boundary(x)), True)

    def test_quadrature_points(self):
        geom = G.HyperRectangle([0, 0], [1, 1])
        x, w = geom.quadrature_points(9, "gausslegd")
        x_true = [-math.sqrt(3 / 5), 0, math.sqrt(3 / 5)]
        x_true = T.tensor(list(itertools.product(x_true, repeat=2)))
        x_true = x_true / 2 + 1 / 2
        w_true = T.tensor([5 / 9, 8 / 9, 5 / 9]) / 2
        w_true = T.prod(
            T.tensor(list(itertools.product(w_true, repeat=2))), axis=-1, keepdims=True
        )
        assert_almost_equal(x, x_true)
        assert_almost_equal(w, w_true)


class TestHyperBall(unittest.TestCase):
    def test_init(self):
        geom = G.HyperBall([0, 0, 0], 1)
        assert_equal(str(geom), "HyperBall")
        assert_equal(geom.center, [0, 0, 0])
        assert_equal(geom.radius, 1)
        assert_equal(geom.bbox[0], [-1, -1, -1])
        assert_equal(geom.bbox[1], [1, 1, 1])
        assert_equal(geom.diam, 2)
        assert_equal(geom.volume, 4 / 3 * T.pi)

    def test_random_points(self):
        geom = G.HyperBall([0, 0, 0], 1)
        x = geom.random_points(100)
        assert_equal(T.shape(x), (100, 3))
        assert_equal(T.all(geom.inside(x)), True)

    def test_random_boundary_points(self):
        geom = G.HyperBall([0, 0, 0], 1)
        x = geom.random_boundary_points(100)
        assert_equal(T.shape(x), (100, 3))
        assert_equal(T.all(geom.on_boundary(x)), True)

    def test_quadrature_points(self):
        geom = G.HyperBall([0, 0, 0], 1)
        x, w = geom.quadrature_points(100, "pseudo")
        assert_equal(T.shape(x), (100, 3))
        assert_equal(T.shape(w), (100, 1))
        assert_equal(T.all(geom.inside(x)), True)


class TestHyperSphere(unittest.TestCase):
    def test_init(self):
        geom = G.HyperSphere([0, 0, 0], 1)
        assert_equal(str(geom), "HyperSphere")
        assert_equal(geom.center, [0, 0, 0])
        assert_equal(geom.radius, 1)
        assert_equal(geom.bbox[0], [-1, -1, -1])
        assert_equal(geom.bbox[1], [1, 1, 1])
        assert_equal(geom.diam, 2)
        assert_equal(geom.area, 4 * T.pi)

    def test_random_points(self):
        geom = G.HyperSphere([0, 0, 0], 1)
        x = geom.random_points(100)
        assert_equal(T.shape(x), (100, 3))
        assert_equal(T.all(geom.inside(x)), True)

    def test_quadrature_points(self):
        geom = G.HyperSphere([0, 0, 0], 1)
        x, w = geom.quadrature_points(110, "lebedev")
        f = T.ones((T.shape(x)[0], 1))
        assert_equal(T.shape(x), (110, 3))
        assert_equal(T.shape(w), (110, 1))
        assert_equal(T.all(geom.inside(x)), True)
        assert_almost_equal(T.transpose(w) @ f, T.tensor([[1]]))


class TestCoordinateTransformers(unittest.TestCase):
    def test_polar_to_cartesian(self):
        polar_coords = T.tensor([[1, 0], [1, T.pi / 2], [1, T.pi], [1, 3 * T.pi / 2]])
        expected_cartesian = T.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])
        cartesian_coords = G.PolarToCartesian.transform(polar_coords)
        assert_almost_equal(cartesian_coords, expected_cartesian, atol=1e-7)

    def test_cartesian_to_polar(self):
        cartesian_coords = T.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])
        expected_polar = T.tensor([[1, 0], [1, T.pi / 2], [1, T.pi], [1, -T.pi / 2]])
        polar_coords = G.CartesianToPolar.transform(cartesian_coords)
        assert_almost_equal(polar_coords, expected_polar, atol=1e-7)

    def test_spherical_to_cartesian(self):
        spherical_coords = T.tensor(
            [
                [1, T.pi / 2, 0],
                [1, T.pi / 2, T.pi / 2],
                [1, T.pi / 2, T.pi],
                [1, T.pi / 2, 3 * T.pi / 2],
            ]
        )
        expected_cartesian = T.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        cartesian_coords = G.SphericalToCartesian.transform(spherical_coords)
        assert_almost_equal(cartesian_coords, expected_cartesian, atol=1e-7)

    def test_cartesian_to_spherical(self):
        cartesian_coords = T.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        expected_spherical = T.tensor(
            [
                [1, T.pi / 2, 0],
                [1, T.pi / 2, T.pi / 2],
                [1, T.pi / 2, T.pi],
                [1, T.pi / 2, -T.pi / 2],
            ]
        )
        spherical_coords = G.CartesianToSpherical.transform(cartesian_coords)
        assert_almost_equal(spherical_coords, expected_spherical, atol=1e-7)

    def test_unit_sphere_angular_to_cartesian(self):
        angular_coords = T.tensor(
            [
                [T.pi / 2, 0],
                [T.pi / 2, T.pi / 2],
                [T.pi / 2, T.pi],
                [T.pi / 2, 3 * T.pi / 2],
            ]
        )
        expected_cartesian = T.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        cartesian_coords = G.UnitSphereAngularToCartesian.transform(angular_coords)
        assert_almost_equal(cartesian_coords, expected_cartesian, atol=1e-7)

    def test_unit_sphere_cartesian_to_angular(self):
        cartesian_coords = T.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        expected_angular = T.tensor(
            [
                [T.pi / 2, 0],
                [T.pi / 2, T.pi / 2],
                [T.pi / 2, T.pi],
                [T.pi / 2, -T.pi / 2],
            ]
        )
        angular_coords = G.UnitSphereCartesianToAngular.transform(cartesian_coords)
        assert_almost_equal(angular_coords, expected_angular, atol=1e-7)

    def test_unit_sphere_angular2_to_cartesian(self):
        angular2_coords = T.tensor(
            [[0, 0], [0, T.pi / 2], [0, T.pi], [0, 3 * T.pi / 2]]
        )
        expected_cartesian = T.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        cartesian_coords = G.UnitSphereAngularToCartesian2.transform(angular2_coords)
        assert_almost_equal(cartesian_coords, expected_cartesian, atol=1e-7)

    def test_unit_sphere_cartesian_to_angular2(self):
        cartesian_coords = T.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        expected_angular2 = T.tensor([[0, 0], [0, T.pi / 2], [0, T.pi], [0, -T.pi / 2]])
        angular2_coords = G.UnitSphereCartesianToAngular2.transform(cartesian_coords)
        assert_almost_equal(angular2_coords, expected_angular2, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
