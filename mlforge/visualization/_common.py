"""Shared plotting helpers. matplotlib is imported lazily so that importing
``mlforge`` never requires it."""

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting needs matplotlib: pip install matplotlib") from exc
    return plt


def prepare_axes(ax=None, figsize=(6, 4.5), projection=None):
    plt = get_plt()
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)
    return ax


def finish(ax, title=None, xlabel=None, ylabel=None, save=None, legend=False):
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False)
    ax.figure.tight_layout()
    if save:
        ax.figure.savefig(save, dpi=140, bbox_inches="tight")
    return ax


def class_colors(n):
    return [PALETTE[i % len(PALETTE)] for i in range(n)]
