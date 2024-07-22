import numpy as np
import scipy.io as spio


def get_numpy_random_state(seed):
    r"""Returns a valid RandomState.

    Parameters
    ----------
    seed : None, int, or np.random.RandomState
        If seed is None, NumPy's global seed is used.

    Returns
    -------
    Valid instance np.random.RandomState

    Notes
    -----
    Inspired by the scikit-learn eponymous function.
    """
    if seed is None:
        return np.random.mtrand._rand

    elif isinstance(seed, int):
        return np.random.RandomState(seed)

    elif isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError("Seed should be None, int or np.random.RandomState")


def savemat(filename, data):
    r"""Interface to `scio.savemat`."""
    spio.savemat(filename, data)


def loadmat(filename):
    r"""This function should be called instead of direct `spio.loadmat`
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    r"""checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries.
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    r"""A recursive function which constructs from matobjects nested
    dictionaries.
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
