try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError as error:
    message = (
        "Fail to import Matplotlib.\n"
        "To visualize with the Matplotlib backend, "
        "you must first install Matplotlib!"
    )
    raise ImportError(message) from error

import os, warnings
import warnings
import itertools
import scipy.io as spio
import numpy as np
from mlbol.utils import GroupedRegistry


backend = GroupedRegistry("matplotlib", group=("A", "M"))


# ---- Configuration ----
default_theme = ["science"]
available_mplstyles = []


def load_mplstyles():
    """Load existing stylesheets stored as '.mplstyle' from `mplstyles`."""
    styles_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mplstyles")
    # reads styles in /styles
    stylesheets = plt.style.core.read_style_directory(styles_path)
    # reads styles in /styles subfolders
    for sub in os.listdir(styles_path):
        sub_path = os.path.join(styles_path, sub)
        if os.path.isdir(sub_path):
            stylesheets.update(plt.style.core.read_style_directory(sub_path))

    available_mplstyles = stylesheets.keys()

    return stylesheets


def register_mplstyles():
    """Register custom stylesheets.

    Copy-paste from:
    https://github.com/matplotlib/matplotlib/blob/a170539a421623bb2967a45a24bb7926e2feb542/lib/matplotlib/style/core.py#L266
    """
    stylesheets = load_mplstyles()
    plt.style.core.update_nested_dict(plt.style.library, stylesheets)
    plt.style.core.available[:] = sorted(plt.style.library.keys())


def register_cmaps():
    """Register custom color maps."""

    sunset = mpl.colors.LinearSegmentedColormap.from_list(
        "sunset",
        (
            # Edit this gradient at https://eltos.github.io/gradient/#sunset=36489A-4A7BB7-6EA6CD-98CAE1-C2E4EF-EAECCC-FEDABB-FDB366-F67E4B-DD3D2D-A50026
            (0.000, (0.212, 0.282, 0.604)),
            (0.100, (0.290, 0.482, 0.718)),
            (0.200, (0.431, 0.651, 0.804)),
            (0.300, (0.596, 0.792, 0.882)),
            (0.400, (0.761, 0.894, 0.937)),
            (0.500, (0.918, 0.925, 0.800)),
            (0.600, (0.996, 0.855, 0.733)),
            (0.700, (0.992, 0.702, 0.400)),
            (0.800, (0.965, 0.494, 0.294)),
            (0.900, (0.867, 0.239, 0.176)),
            (1.000, (0.647, 0.000, 0.149)),
        ),
    )
    nightfall = mpl.colors.LinearSegmentedColormap.from_list(
        "nightfall",
        (
            # Edit this gradient at https://eltos.github.io/gradient/#sunset=125A56-00767B-238F9D-42A7C6-60BCE9-9DCCEF-C6DBED-DEE6E7-ECEADA-FDE6B2-F9D576-FFB954-FD9A44-F57634-E94C1F-D11807-A01813
            (0.000, (0.071, 0.353, 0.337)),
            (0.063, (0.000, 0.463, 0.482)),
            (0.125, (0.137, 0.561, 0.616)),
            (0.188, (0.259, 0.655, 0.776)),
            (0.250, (0.376, 0.737, 0.914)),
            (0.313, (0.616, 0.800, 0.937)),
            (0.375, (0.776, 0.859, 0.929)),
            (0.438, (0.871, 0.902, 0.906)),
            (0.500, (0.925, 0.918, 0.855)),
            (0.563, (0.992, 0.902, 0.698)),
            (0.625, (0.976, 0.835, 0.463)),
            (0.688, (1.000, 0.725, 0.329)),
            (0.750, (0.992, 0.604, 0.267)),
            (0.813, (0.961, 0.463, 0.204)),
            (0.875, (0.914, 0.298, 0.122)),
            (0.938, (0.820, 0.094, 0.027)),
            (1.000, (0.627, 0.094, 0.075)),
        ),
    )

    mpl.colormaps.register(sunset)
    mpl.colormaps.register(nightfall)
    # plt.register_cmap("sunset", sunset)
    # plt.register_cmap("nightfall", nightfall)


register_mplstyles()
register_cmaps()


def set_theme(theme=None):
    if theme is None:
        plt.style.use(default_theme)
    else:
        if isinstance(theme, str):
            theme = [theme]
        valid_style = []
        for st in theme:
            if st not in available_mplstyles:
                msg = (
                    f"style should be one of {''.join(map(repr, available_mplstyles))}"
                    f", but got {st}."
                )
                warnings.warn(msg, UserWarning)
            else:
                valid_style.append(st)
        plt.style.use(valid_style)


# ---- Attributes ----
backend.register_("A", "plt", plt)


# ---- Methods ----
@backend.register("M")
def multi_plot(xs, ys, labels=None, outfile=None, style=None, **kwargs):
    if labels is None:
        labels = [f"Label {i}" for i in range(len(xs))]
    if style is None:
        style = ["science", "jcp-sc-line", "cmp1d"]

    with plt.style.context(style):
        fig, ax = plt.subplots()
        for x, y, label in zip(xs, ys, labels):
            ax.plot(x, y, label=label, clip_on=False)
        ax.legend()
        ax.autoscale(tight=True)
        ax.set(**kwargs)

        # save figure
        if outfile:
            fig.savefig(outfile)
        else:
            plt.show()
            plt.close()


@backend.register("M")
def multi_plot_by_files(
    infiles, key_x="x", key_y="y", labels=None, outfile=None, style=None, **kwargs
):
    xs, ys = [], []
    if labels:
        new_labels = []
        for file, label in zip(infiles, labels):
            data = spio.loadmat(file) if os.path.exists(file) else None
            if data is not None:
                xs.append(np.ravel(data[key_x]))
                ys.append(np.ravel(data[key_y]))
                new_labels.append(label)
            else:
                warnings.warn(f"{file} does not exist, skip.")
        labels = new_labels
    else:
        for file in infiles:
            data = spio.loadmat(file) if os.path.exists(file) else None
            if data is not None:
                xs.append(np.ravel(data[key_x]))
                ys.append(np.ravel(data[key_y]))
            else:
                warnings.warn(f"{file} does not exist, skip.")

    multi_plot(xs, ys, labels=labels, outfile=outfile, style=style, **kwargs)


@backend.register("M")
def multi_imshow(
    Xs, titles=None, outfile=None, bbox=None, log_cmap=False, style=None, **kwargs
):
    m, n = len(Xs), len(Xs[0])
    if titles is None:
        titles = [[f"Title {i*n+j}" for j in range(n)] for i in range(m)]
    if style is None:
        style = ["science", "jcp-dc-plain", "im2d"]

    with plt.style.context(style):
        fig, axes = plt.subplots(m, n)
        if m == 1 and n == 1:
            axes = [[axes]]
        else:
            axes = axes.reshape(m, n)
        for i, j in itertools.product(range(m), range(n)):
            X, ax, title = Xs[i][j], axes[i][j], titles[i][j]
            cmap = plt.get_cmap("sunset")
            cmap.set_under(color="gray")
            if log_cmap:
                clim = [-7, 0]
                X_old = X
                X = np.log10(np.maximum(X, 1e-8))
                X = np.maximum(X, -7)
                X[X_old < 0] = clim[0] - (clim[1] - clim[0]) / 254
            if bbox is not None:
                extent = []
                for d in range(len(bbox[0])):
                    extent.append(bbox[0][d])
                    extent.append(bbox[1][d])
            else:
                extent = None
            vmin, vmax = X.min(), X.max()
            im = ax.imshow(
                X.T,
                extent=extent,
                cmap="sunset",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set(**kwargs)
            if title:
                ax.set_title(title)
            ax.autoscale(tight=True)
        pos_r0 = list(axes[0][-1].get_position().bounds)
        pos_r1 = list(axes[-1][-1].get_position().bounds)
        bottom = pos_r1[1]
        height = pos_r0[1] + pos_r0[3] - pos_r1[1]
        cax = fig.add_axes([0.95, bottom, 0.025, height])
        fig.colorbar(im, cax=cax)
        fig.tight_layout(rect=[0, 0, 0.95, 1])

        # save figure
        if outfile:
            fig.savefig(outfile)
        else:
            plt.show()
            plt.close()


@backend.register("M")
def multi_imshow_by_files(
    infiles,
    key_X="X",
    titles=None,
    outfile=None,
    bbox=None,
    log_cmap=False,
    style=None,
    **kwargs,
):
    m, n = len(infiles), len(infiles[0])
    Xs = [[]]
    if titles:
        new_titles = [[]]
        for i in range(m):
            for j in range(n):
                file, title = infiles[i][j], titles[i][j]
                data = spio.loadmat(file) if os.path.exists(file) else None
                if bbox is None:
                    bbox = data.get("bbox", None)
                if data is not None:
                    Xs[i].append(data[key_X])
                    new_titles[i].append(title)
                else:
                    warnings.warn(f"{file} does not exist, skip.")
        titles = new_titles
    else:
        for i in range(m):
            for j in range(n):
                file = infiles[i][j]
                data = spio.loadmat(file) if os.path.exists(file) else None
                if bbox is None:
                    bbox = data.get("bbox", None)
                if data is not None:
                    Xs[i].append(data[key_X])
                else:
                    warnings.warn(f"{file} does not exist, skip.")

    multi_imshow(
        Xs,
        titles=titles,
        outfile=outfile,
        bbox=bbox,
        log_cmap=log_cmap,
        style=style,
        **kwargs,
    )


