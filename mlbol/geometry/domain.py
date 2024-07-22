from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Iterable
from typing import Union
from itertools import accumulate
from mlbol.dtensor import Tensor
from mlbol.dtensor import concatenate
from mlbol.dtensor import gkron
from mlbol.geometry.base import Geometry

__all__ = ["Domain"]


class Domain(OrderedDict[str, Geometry]):
    """Base class for a domain.

    A domain is a ordered dictionary of geometries. The main functionalities are:

    * draw points within the domain
    * draw points on the boundary of domain
    """

    @property
    def dims(self) -> Dict[str, Tuple[int, int]]:
        cumdims = list(accumulate([0] + list(v.ndim for v in self.values())))
        result = {
            key: (cumdims[i], cumdims[i + 1]) for i, key in enumerate(self.keys())
        }
        return result

    def points(
        self,
        n: Union[int, Dict[str, int]],
        mode: Union[str, Dict[str, str]],
        config: Dict[str, Dict[str, Any]] = {},
    ) -> Tensor:
        """Draw points within the domain.

        Parameters
        ----------
        n : int or dict[str, int]
            Number of points.
        mode : str or dict[str, str], optional
            Generator type. Must be 'random' or 'uniform'. Default is 'random'.
        config : dict[str, dict[str, Any]], optional
            Generator arguments. Default is `{}`.

        Returns
        -------
        Tensor
            If `n` is an int, the output tensor is of shape (n, d1+...+dk).
            If `n` is a tuple of ints `(n1,...,nk)`, the output tensor
            is of shape (n1*...*nk, d1+...+dk).
        """
        product = (isinstance(n, Iterable)) and (isinstance(mode, Iterable))
        if isinstance(n, int):
            n = {k: n for k in self.keys()}
        if isinstance(mode, str):
            mode = {k: mode for k in self.keys()}
        xs = []
        for i, g in self.items():
            m, d, a = n[i], mode[i], config.get(i, {})
            if d == "random":
                x = g.random_points(m, **a)
            if d == "uniform":
                x = g.uniform_points(m, **a)
            xs.append(x)
        x = concatenate(gkron(xs) if product else xs, axis=-1)
        return x

    def boundary_points(
        self,
        k: str,
        n: Dict[str, int],
        mode: Dict[str, str],
        config: Dict[str, Dict[str, Any]] = {},
    ) -> Tensor:
        """Draw points on the boundary of domain.

        Parameters
        ----------
        k : str
            Key of geometry in which boundary is selected.
        n : dict[str, int]
            Number of points within the geometry or at the boundary.
        mode : dict[str, str]
            Generator mode.
        config : dict[str, dict[str, Any]], optional
            Generator arguments. Default is `{}`.

        Returns
        -------
        Tensor
            If `n = (n_1,...,n_k)`, then output tensor is of shape
            `(n_0*n_1*...*n_k, D)`.
        """
        if k not in self:
            raise ValueError(f"Fail to find a geometry named '{k}'.")
        X = OrderedDict({})
        for i, g in self.items():
            m, d, a = n[i], mode[i], config.get(i, {})
            if i == k:
                if d == "random":
                    X.update({i: g.random_boundary_points(m, **a)})
                if d == "uniform":
                    X.update({i: g.uniform_boundary_points(m, **a)})
            else:
                if d == "random":
                    X.update({i: g.random_points(m, **a)})
                if d == "uniform":
                    X.update({i: g.uniform_points(m, **a)})
        X = list(X.values())
        x = concatenate(gkron(X), axis=-1)
        return x

    def split_as_dict(self, x: Tensor) -> Dict[str, Tensor]:
        out = {}
        n = 0
        for k, v in self.items():
            out.update({k: x[:, n : n + v.ndim]})
            n += v.ndim
        return out
