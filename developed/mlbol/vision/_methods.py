from typing import Any
from typing import TypeVar
from mlbol.vision._engine import _VisionEngine as _vision_engine

TensorLike = TypeVar("TensorLike")

def _apply_method(name: str, *args, **kwargs) -> Any:
    for interface in [_vision_engine]:
        if hasattr(interface, name):
            return getattr(interface, name)(*args, **kwargs)
    raise ValueError(f"Method {name} is not available.")


def multi_plot(
    xs: list[TensorLike],
    ys: list[TensorLike],
    labels: list[str] = None,
    outfile: str | None = None,
    style: list[str] | None = None,
    **kwargs,
) -> None:
    """Plot multiple lines in one figure.

    Parameters
    ----------
    xs : list of tensor_like
        Horizontal coordinates. Each element should be a 1D array.
    ys : list of tensor_like
        Vertical coordinates. Each element should be a 1D array.
    labels : list of strs or None, optional
        Each element represents a label for each line.
        If None (default), lineplots are named as `Label n`.
    outfile : str or None, optional
        If None (default), interative mode.
    style : list of sts or None, optional
        If None (default), use default lineplot style.
    """
    return _apply_method(
        "multi_plot", xs, ys, labels=labels, outfile=outfile, style=style, **kwargs
    )


def multi_plot_by_files(
    infiles: list[str],
    key_x: str = "x",
    key_y: str = "y",
    labels: list[str] | None = None,
    outfile: str | None = None,
    style: list[str] | None = None,
    **kwargs,
):
    """Plot multiple lines in one figure after reading data from files.

    Each file store a dictionary whose `key_x` represents horizontal coordinates,
    and `key_y` represents vertical coordinates.

    Parameters
    ----------
    infiles: list of strs
        Input files.
    key_x : str
        Key for horizontal coordinates.
    key_y : str
        Key for vertical coordinates.
    labels : list of strs or None, optional
        Each element represents a label for each line.
        If None (default), lineplots are named as `Label n`.
    outfile : str or None, optional
        If None (default), interative mode.
    style : list of sts or None, optional
        If None (default), use default lineplot style.
    """
    return _apply_method(
        "multi_plot_by_files",
        infiles,
        key_x=key_x,
        key_y=key_y,
        labels=labels,
        outfile=outfile,
        style=style,
        **kwargs,
    )


def multi_imshow(
    Xs: list[TensorLike],
    titles: list[str] = None,
    outfile: str = None,
    bbox: list[list[float]] | None = None,
    log_cmap: bool = False,
    style: list[str] | None = None,
    **kwargs,
):
    """Display a list of data as images, i.e., on 2D regular rasters.

    The input should either be actual RGB data, or 2D scalar data, which
    will be rendered as a pseudocolor image. For displaying a grayscale
    image, set up the colormapping using the parameters `cmap='gray',
    vmin=0, vmax=255`.

    Parameters
    ----------
    Xs : list[_TensorLike]
        The image data.
    titles : list[str] or None, optional
        Title for each subplot. Default is None.
    outfile : str, optional
        If None (default), interative mode.
    bbox : list[list[float]] | None, optional
        Bounding box. Default is None.
    log_cmap : bool, optional
        Determine whether use logarithm scale on color map.
    style : list[str] | None, optional
        If None (default), use imshow style with 'sunset' colormap.
    """
    return _apply_method(
        "multi_imshow",
        Xs,
        titles=titles,
        outfile=outfile,
        bbox=bbox,
        log_cmap=log_cmap,
        style=style,
        **kwargs,
    )


def multi_imshow_by_files(
    infiles: list[str],
    key_X: str = "X",
    titles: list[str] = None,
    outfile: str = None,
    bbox: list[list[float]] | None = None,
    log_cmap: bool = False,
    style: list[str] | None = None,
    **kwargs,
):
    """Display a list of data as images, i.e., on 2D regular rasters.

    The input should either be actual RGB data, or 2D scalar data, which
    will be rendered as a pseudocolor image. For displaying a grayscale
    image, set up the colormapping using the parameters `cmap='gray',
    vmin=0, vmax=255`.

    Parameters
    ----------
    infiles : list[str]
        Input files.
    key_X : str, optional
        Key of image data. Default is 'X'.
    titles : list[str] or None, optional
        Title for each subplot. Default is None.
    outfile : str, optional
        If None (default), interative mode.
    bbox : list[list[float]] | None, optional
        Bounding box. Default is None.
    log_cmap : bool, optional
        Determine whether use logarithm scale on color map.
    style : list[str] | None, optional
        If None (default), use imshow style with 'sunset' colormap.
    """
    return _apply_method(
        "multi_imshow_by_files",
        infiles,
        key_X=key_X,
        titles=titles,
        outfile=outfile,
        bbox=bbox,
        log_cmap=log_cmap,
        style=style,
        **kwargs,
    )