import logging
import numpy as np
import h5py
import scipy.io as matpy
from typing import Any
from typing import Dict
from typing import Union
from mlbol.dtensor import Tensor
from mlbol.dtensor import to_numpy as _to_numpy
from mlbol.dtensor import tensor as _tensor
from mlbol.utils.debug.format import add_suffix as _add_suffix

try:
    from optree import tree_map as _tree_map
except:
    from mlbol.utils.tree._pytree import tree_map as _tree_map

__all__ = [
    "HDF5_SUFFIX",
    "MAT_SUFFIX",
    "NPY_SUFFIX",
    "save_hdf5",
    "load_hdf5",
    "save_mat",
    "load_mat",
    "save_npy",
    "load_npy",
]

HDF5_SUFFIX = (".h5", ".hdf5")
MAT_SUFFIX = (".mat",)
NPY_SUFFIX = (".npy",)

_DATA_TREE = Any
_DATA_NODE = Any
_MAT_TYPE = Any
_MAT_STRUCT = matpy.matlab.mat_struct



def _on_save_start(data: Tensor) -> np.ndarray:
    return _to_numpy(data)


def _on_load_end(data: _DATA_NODE) -> Union[Tensor, _DATA_NODE]:
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        return _tensor(data)
    return data


def _to_datatree(dict: Dict[str, Any]) -> _DATA_TREE:
    r"""checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries.
    """
    for key in dict:
        if isinstance(dict[key], _MAT_STRUCT):
            dict[key] = _matobj_to_datatree(dict[key])
    return dict


def _matobj_to_datatree(matobj: _MAT_TYPE) -> _DATA_TREE:
    r"""A recursive function which constructs from matobjects nested
    dictionaries.
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, _MAT_STRUCT):
            dict[strg] = _matobj_to_datatree(elem)
        else:
            dict[strg] = elem
    return dict


def _save_mat_impl(file: str, data: _DATA_TREE) -> None:
    r"""Interface to `scipy.io.savemat`."""
    matpy.savemat(file, data)


def _load_mat_impl(file: str) -> _DATA_TREE:
    r"""This function should be called instead of direct `scipy.io.loadmat`
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    """
    data = matpy.loadmat(file, struct_as_record=False, squeeze_me=True)
    return _to_datatree(data)


def save_hdf5(path: str, data: Dict[str, _DATA_TREE]) -> None:
    """Save data to a local HDF5 file.

    Parameters
    ----------
    path : str
        Output path.
    data : Dict[str, data_tree]
        A dictionary of data trees to store.
    """
    try:
        path = _add_suffix(path, HDF5_SUFFIX)
        with h5py.File(path, "w") as hdf5_file:
            for key, value in data.items():
                value = _tree_map(_on_save_start, value)
                hdf5_file.create_dataset(key, data=value)
    except Exception as exc:
        logging.error(
            f"{save_hdf5.__name__}: Exception happens when saving file '{path}'."
        )
        raise


def load_hdf5(path: str) -> Dict[str, _DATA_TREE]:
    """Load data from a local HDF5 file.

    Parameters
    ----------
    path : str
        Input path.

    Returns
    -------
    data : Dict[str, data_tree]
        A dictionary of data trees loaded from local file.
    """
    try:
        data = {}
        with h5py.File(path, "r") as hdf5_file:
            for key in hdf5_file.keys():
                data[key] = _tree_map(_on_load_end, hdf5_file[key][()])
    except FileNotFoundError as err:
        logging.error(f"{load_hdf5.__name__}: File '{path}' does not exist.")
        raise
    except Exception as exc:
        logging.error(
            f"{load_hdf5.__name__}: Exception happens when loading file '{path}'."
        )
        raise
    return data


def save_mat(path: str, data: _DATA_TREE) -> None:
    """Save data to a local MAT file.

    Parameters
    ----------
    path : str
        Output path.
    data : data_tree
        Data tree to store.
    """
    try:
        path = _add_suffix(path, MAT_SUFFIX)
        data = _tree_map(_on_save_start, data)
        _save_mat_impl(path, data)
    except Exception as exc:
        logging.error(
            f"{save_mat.__name__}: Exception happens when saving file '{path}'."
        )
        raise


def load_mat(path: str, remove_info: bool = True) -> _DATA_TREE:
    """Load data from a local MAT file.

    Parameters
    ----------
    path : str
        Input path.
    remove_info : bool, default is True
        If True, remove all the auxiliary information generated
        when loading a MAT file.

    Returns
    -------
    data : Dict[str, data_tree]
        A data tree loaded from a local MAT file.
    """
    try:
        data = _load_mat_impl(path)
        data = _tree_map(_on_load_end, data)
        if remove_info:
            for k in ["__header__", "__version__", "__globals__"]:
                data.pop(k)
    except FileNotFoundError as err:
        logging.error(f"{load_mat.__name__}: File '{path}' does not exist.")
        raise
    except Exception as exc:
        logging.error(
            f"{load_mat.__name__}: Exception happens when loading file '{path}'."
        )
        raise
    return data


def save_npy(path: str, data: _DATA_TREE) -> None:
    """Save data to a local NPY file.

    Parameters
    ----------
    path: str
        Output path.
    data : data_tree
        Data tree to store.
    """
    try:
        path = _add_suffix(path, NPY_SUFFIX)
        data = _tree_map(_on_save_start, data)
        np.save(path, data, allow_pickle=True)
    except Exception as exc:
        logging.error(
            f"{save_npy.__name__}: Exception happens when saving file '{path}'."
        )
        raise


def load_npy(path: str) -> _DATA_TREE:
    """Load data from a local NPY file.

    Parameters
    ----------
    path : str
        Input path.

    Returns
    -------
    data : data_tree
        A data tree loaded from a local NPY file.
    """
    try:
        data = np.load(path, allow_pickle=True)
        data = _tree_map(_on_load_end, data)
    except FileNotFoundError as err:
        logging.error(f"{load_npy.__name__}: File '{path}' does not exist.")
        raise
    except Exception as exc:
        logging.error(
            f"{load_npy.__name__}: Exception happens when loading file '{path}'."
        )
        raise
    return data
