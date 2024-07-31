from abc import ABC
from abc import abstractmethod
from typing import Callable
from mlbol.dtensor import Tensor
from mlbol.dtensor import as_tensor
from mlbol.dtensor import sin
from mlbol.dtensor import cos
from mlbol.dtensor import stack
from mlbol.dtensor import zeros
from mlbol.dtensor import shape
from mlbol.dtensor import sqrt
from mlbol.dtensor import arccos
from mlbol.dtensor import arctan2

__all__ = [
    "CoordinateTransform",
    "PolarToCartesian",
    "CatesianToPolar",
    "SphericalToCartesian",
    "CartesianToSpherical",
    "UnitSphereAngularToCartesian",
    "UnitSphereCartesianToAngular",
    "UnitSphereAngularToCartesian2",
    "UnitSphereCartesianToAngular2",
]


class CoordinateTransform(ABC):
    @classmethod
    def transform(cls, coordinates: Tensor) -> Tensor:
        return cls._apply(cls._transform_impl, coordinates)

    @classmethod
    def jacobian(cls, coordinates: Tensor) -> Tensor:
        return cls._apply(cls._jacobian_impl, coordinates)

    @staticmethod
    def _apply(func: Callable, coordinates: Tensor) -> Tensor:
        coordinates = as_tensor(coordinates)
        return func(coordinates)

    @staticmethod
    @abstractmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        pass

    @staticmethod
    @abstractmethod
    def _jacobian_impl(coordinates: Tensor) -> Tensor:
        pass


class PolarToCartesian(CoordinateTransform):
    @staticmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        r, p = coordinates[..., 0], coordinates[..., 1]
        x = r * cos(p)
        y = r * sin(p)
        return stack((x, y), axis=-1)

    @staticmethod
    def _jacobian_impl(coordinates: Tensor) -> Tensor:
        r, p = coordinates[..., 0], coordinates[..., 1]
        J = zeros((*shape(coordinates)[:-1], 2, 2))
        J[..., 0, 0] = cos(p)
        J[..., 0, 1] = -r * sin(p)
        J[..., 1, 0] = sin(p)
        J[..., 1, 1] = r * cos(p)
        return J


class CartesianToPolar(CoordinateTransform):
    @staticmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        x, y = coordinates[..., 0], coordinates[..., 1]
        r = sqrt(x**2 + y**2)
        p = arctan2(y, x)
        return stack((r, p), axis=-1)

    @staticmethod
    def _jacobian_impl(coordinates: Tensor) -> Tensor:
        x, y = coordinates[..., 0], coordinates[..., 1]
        r = sqrt(x**2 + y**2)
        J = zeros((*shape(coordinates)[:-1], 2, 2))
        J[..., 0, 0] = x / r
        J[..., 0, 1] = y / r
        J[..., 1, 0] = -y / (x**2 + y**2)
        J[..., 1, 1] = x / (x**2 + y**2)
        return J


class SphericalToCartesian(CoordinateTransform):
    @staticmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        r, p, a = (
            coordinates[..., 0],
            coordinates[..., 1],
            coordinates[..., 2],
        )
        x = r * sin(p) * cos(a)
        y = r * sin(p) * sin(a)
        z = r * cos(p)
        return stack((x, y, z), axis=-1)

    @staticmethod
    def _jacobian_impl(coordinates: Tensor) -> Tensor:
        r, p, a = (
            coordinates[..., 0],
            coordinates[..., 1],
            coordinates[..., 2],
        )
        J = zeros((*shape(coordinates)[:-1], 3, 3))
        J[..., 0, 0] = sin(p) * cos(a)
        J[..., 0, 1] = r * cos(p) * cos(a)
        J[..., 0, 2] = -r * sin(p) * sin(a)
        J[..., 1, 0] = sin(p) * sin(a)
        J[..., 1, 1] = r * cos(p) * sin(a)
        J[..., 1, 2] = r * sin(p) * cos(a)
        J[..., 2, 0] = cos(p)
        J[..., 2, 1] = -r * sin(p)
        J[..., 2, 2] = 0
        return J


class CartesianToSpherical(CoordinateTransform):
    @staticmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        r = sqrt(x**2 + y**2 + z**2)
        p = arccos(z / r)
        a = arctan2(y, x)
        return stack((r, p, a), axis=-1)

    @staticmethod
    def _jacobian_impl(coordinates: Tensor) -> Tensor:
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        r = sqrt(x**2 + y**2 + z**2)
        J = zeros((*shape(coordinates)[:-1], 3, 3))
        J[..., 0, 0] = x / r
        J[..., 0, 1] = y / r
        J[..., 0, 2] = z / r
        J[..., 1, 0] = x * z / (r**2 * sqrt(x**2 + y**2))
        J[..., 1, 1] = y * z / (r**2 * sqrt(x**2 + y**2))
        J[..., 1, 2] = -sqrt(x**2 + y**2) / r**2
        J[..., 2, 0] = -y / (x**2 + y**2)
        J[..., 2, 1] = x / (x**2 + y**2)
        J[..., 2, 2] = 0
        return J


class UnitSphereAngularToCartesian(CoordinateTransform):
    @staticmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        p, a = coordinates[..., 0], coordinates[..., 1]
        x = sin(p) * cos(a)
        y = sin(p) * sin(a)
        z = cos(p)
        return stack((x, y, z), axis=-1)

    @staticmethod
    def _jacobian_impl(coordinates: Tensor) -> Tensor:
        p, a = coordinates[..., 0], coordinates[..., 1]
        J = zeros((*shape(coordinates)[:-1], 3, 2))
        J[..., 0, 0] = cos(p) * cos(a)
        J[..., 0, 1] = -sin(p) * sin(a)
        J[..., 1, 0] = cos(p) * sin(a)
        J[..., 1, 1] = sin(p) * cos(a)
        J[..., 2, 0] = -sin(p)
        J[..., 2, 1] = 0
        return J


class UnitSphereCartesianToAngular(CoordinateTransform):
    @staticmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        p = arccos(z / sqrt(x**2 + y**2 + z**2))
        a = arctan2(y, x)
        return stack((p, a), axis=-1)

    @staticmethod
    def _jacobian_impl(coordinates):
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        r = sqrt(x**2 + y**2 + z**2)
        J = zeros((*shape(coordinates)[:-1], 2, 3))
        J[..., 0, 0] = x * z / (r**2 * sqrt(x**2 + y**2))
        J[..., 0, 1] = y * z / (r**2 * sqrt(x**2 + y**2))
        J[..., 0, 2] = -sqrt(x**2 + y**2) / r**2
        J[..., 1, 0] = -y / (x**2 + y**2)
        J[..., 1, 1] = x / (x**2 + y**2)
        J[..., 1, 2] = 0
        return J


class UnitSphereAngularToCartesian2(CoordinateTransform):
    @staticmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        m, a = coordinates[..., 0], coordinates[..., 1]
        x = sqrt(1 - m**2) * cos(a)
        y = sqrt(1 - m**2) * sin(a)
        z = m
        return stack((x, y, z), axis=-1)

    @staticmethod
    def _jacobian_impl(coordinates: Tensor) -> Tensor:
        m, a = coordinates[..., 0], coordinates[..., 1]
        J = zeros((*shape(coordinates)[:-1], 2, 3))
        J[..., 0, 0] = -m / (sqrt(1 - m**2)) * cos(a)
        J[..., 0, 1] = -m / (sqrt(1 - m**2)) * sin(a)
        J[..., 0, 2] = 1
        J[..., 1, 0] = -sqrt(1 - m**2) * sin(a)
        J[..., 1, 1] = sqrt(1 - m**2) * cos(a)
        J[..., 1, 2] = 0
        return J


class UnitSphereCartesianToAngular2(CoordinateTransform):
    @staticmethod
    def _transform_impl(coordinates: Tensor) -> Tensor:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        m = z
        a = arctan2(y, x)
        return stack((m, a), axis=-1)

    @staticmethod
    def _jacobian_impl(coordinates: Tensor) -> Tensor:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        J = zeros((*shape(coordinates)[:-1], 3, 2))
        J[..., 0, 0] = -y / (x**2 + y**2)
        J[..., 0, 1] = 0
        J[..., 1, 0] = x / (x**2 + y**2)
        J[..., 1, 1] = 0
        J[..., 2, 0] = 0
        J[..., 2, 1] = 1
        return J
