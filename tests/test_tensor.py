import unittest
import numpy as np

try:
    import torch
except ImportError as error:
    message = (
        "Impossible to import PyTorch.\n"
        "To use DeepMLT with the PyTorch backend, "
        "you must first install PyTorch!"
    )
    raise ImportError(message) from error

import mlbol as dl
from mlbol import dtensor as T
from tests.utils import assert_, assert_equal


class TestNumpyBackend(unittest.TestCase):
    def test_types(self):
        dl.set_backend("numpy")
        assert_equal(T.int32, np.int32)
        assert_equal(T.int64, np.int64)
        assert_equal(T.float32, np.float32)
        assert_equal(T.float64, np.float64)
        assert_equal(T.complex64, np.complex64)
        assert_equal(T.complex128, np.complex128)
        assert_equal(T.e, np.e)
        assert_equal(T.pi, np.pi)
        assert_equal(T.inf, np.inf)
        assert_equal(np.isnan(T.nan), True)

    def test_tensor_basics(self):
        dl.set_backend("numpy")
        a = [1, 2, 3]
        assert_equal(T.is_tensor(a), False)
        b = T.tensor(a, dtype=T.float64)
        assert_equal(T.is_tensor(b), True)
        assert_equal(T.context(b), {"dtype": np.float64})

        c = T.as_tensor(a, dtype=T.float32)
        assert_equal(T.is_tensor(c), True)
        assert_equal(T.context(c), {"dtype": np.float32})
        assert_equal(isinstance(T.to_numpy(c), np.ndarray), True)
        assert_equal(T.copy(c), c)

        d = T.empty((10, 2), dtype=T.float32)
        d_ = T.empty_like(d)
        assert_equal(T.size(d, 0), 10)
        assert_equal(T.shape(d), (10, 2))
        assert_equal(T.shape(d_), T.shape(d))
        assert_equal(T.context(d_), T.context(d))

        d = T.zeros((10, 2), dtype=T.float32)
        d_ = T.zeros_like(d)
        assert_equal(T.size(d, 0), 10)
        assert_equal(T.shape(d), (10, 2))
        assert_equal(T.shape(d_), T.shape(d))
        assert_equal(T.context(d_), T.context(d))
        assert_equal(d, 0)

        d = T.ones((10, 2), dtype=T.float32)
        d_ = T.ones_like(d)
        assert_equal(T.size(d, 0), 10)
        assert_equal(T.shape(d), (10, 2))
        assert_equal(T.shape(d_), T.shape(d))
        assert_equal(T.context(d_), T.context(d))
        assert_equal(d, 1)

        e = T.rand((10, 2), seed=1)
        assert_equal(T.shape(e), (10, 2))
        e = T.randn((4, 3), seed=1)
        assert_equal(T.shape(e), (4, 3))
        e = T.gamma(range(5), seed=1)
        assert_equal(T.shape(e), (5,))
        e = T.uniform(low=0.0, high=1.0, size=(3, 4), seed=1)
        assert_equal(T.shape(e), (3, 4))
        e = T.normal(loc=0.0, scale=1.0, size=(3, 4), seed=1)
        assert_equal(T.shape(e), (3, 4))
        e = T.glorot_uniform((3, 4), scale=1, mode="fan_avg", seed=1)
        assert_equal(T.shape(e), (3, 4))
        e = T.glorot_normal((3, 4), scale=1, mode="fan_avg", truncated=True, seed=1)
        assert_equal(T.shape(e), (3, 4))
        e = T.eye(4)
        assert_equal(e, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        e = T.diag(range(3), k=-1)
        assert_equal(e, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0]])
        e = T.arange(0, 10, 2)
        assert_equal(e, [0, 2, 4, 6, 8])
        e = T.linspace(0, 5, 6)
        assert_equal(e, [0, 1, 2, 3, 4, 5])
        f1, f2 = T.meshgrid(T.arange(3), T.arange(2), indexing="xy")
        assert_equal(f1, [[0, 1, 2], [0, 1, 2]])
        assert_equal(f2, [[0, 0, 0], [1, 1, 1]])

        g = T.arange(24).reshape(3, 4, 2)
        assert_equal(T.shape(g), (3, 4, 2))
        g = T.moveaxis(g, 0, 1)
        assert_equal(T.shape(g), (4, 3, 2))
        g = T.transpose(g)
        assert_equal(T.shape(g), (2, 3, 4))
        g = T.swapaxes(g, 1, 2)
        assert_equal(T.shape(g), (2, 4, 3))
        g = T.concatenate([g, g], axis=0)
        assert_equal(T.shape(g), (4, 4, 3))
        g = T.stack([g, g], axis=0)
        assert_equal(T.shape(g), (2, 4, 4, 3))
        g_ = T.flip(g, axis=0)
        assert_equal(g[:, ...], g_[::-1, ...])

        h = T.sort(T.rand((10,)))
        assert_equal(h[:-1] < h[1:], True)
        h = T.count_nonzero(T.tensor([0, 0, 0, 1, 2]))
        assert_equal(h, 2)
        assert_equal(T.all([-1, 4, 5]), True)
        assert_equal(T.any([[True, False], [False, False]]), True)
        h = T.arange(10)
        assert_equal(T.where(h < 5, h, 10 * h), [0, 1, 2, 3, 4, 50, 60, 70, 80, 90])


class TestHybridBackend(unittest.TestCase):
    def test_set_backend(self):
        toplevel_backend = dl.get_dtensor_backend()

        # Set in context manager
        with dl.dtensor.dtensor_profile("numpy"):
            assert_equal(dl.get_dtensor_backend(), "numpy")
            assert_(isinstance(T.tensor([1, 2, 3]), np.ndarray), True)
            assert_(isinstance(T.tensor([1, 2, 3]), np.ndarray), True)
            assert_(T.float32 is np.float32, True)

            with dl.dtensor.dtensor_profile("pytorch"):
                assert_equal(dl.get_dtensor_backend(), "pytorch")
                assert_(torch.is_tensor(T.tensor([1, 2, 3])))
                assert_(torch.is_tensor(T.tensor([1, 2, 3])))
                assert_(T.float32 is torch.float32)
                m = T.ravel_multi_index(
                    ([1, 3], [-1, 1], [1, 0]), (3, 4, 2), mode="wrap", order="C"
                )
                assert_equal(m, torch.tensor([15, 2], dtype=torch.long))
                m = T.ravel_multi_index(
                    ([1, 3], [-1, 1], [1, 0]), (3, 4, 2), mode="clip", order="C"
                )
                assert_equal(m, torch.tensor([9, 18], dtype=torch.long))
                i = T.unravel_index(22, (7, 6))
                assert_equal(
                    i,
                    (
                        torch.tensor([3], dtype=torch.long),
                        torch.tensor([4], dtype=torch.long),
                    ),
                )
                i = T.unravel_index([22, 41, 37], (77,))
                assert_equal(i[0], torch.tensor([22, 41, 37], dtype=torch.long))
                a, b, n = (1, 0, 0), (3, 3, 2), (3, 4, 2)
                r = T.ravel_multi_range(a, b, n, mode="raise", order="C")
                assert_equal(
                    r,
                    torch.tensor(
                        [8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21], dtype=torch.long
                    ),
                )

            # Sets back to numpy
            assert_equal(dl.get_dtensor_backend(), "numpy")
            assert_(isinstance(T.tensor([1, 2, 3]), np.ndarray))
            assert_(isinstance(T.tensor([1, 2, 3]), np.ndarray))
            assert_(T.float32 is np.float32)

        # Set not in context manager
        dl.set_backend("pytorch")
        assert_equal(dl.get_dtensor_backend(), "pytorch")
        dl.set_backend(toplevel_backend)

        assert_equal(dl.get_dtensor_backend(), toplevel_backend)

        # Improper name doesn't reset backend
        # try:
        #     dl.set_backend("not-a-real-backend")
        # except:
        #     assert_equal(T._get_registry(), toplevel_backend)
        #     raise ValueError(f"Fail to switch backend from `{dl.get_dtensor_backend()}` to `not-a-real-backend`.")


if __name__ == "__main__":
    unittest.main()
