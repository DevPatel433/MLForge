"""GIF animations of learning processes (need ``Pillow``, a matplotlib dependency)."""
import numpy as np

from ..optimization import minimize
from ._common import PALETTE, get_plt


def animate_kmeans(model, X, path="kmeans.gif", fps=3, title="K-Means: centroid movement"):
    """Animate the centroid trajectory stored in ``model.history``."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    from ..utils.pairwise import pairwise_distances
    plt = get_plt()
    X = np.asarray(X)
    hist = np.array(model.history)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    def draw(frame):
        ax.clear()
        centers = hist[frame]
        labels = pairwise_distances(X, centers, "sqeuclidean").argmin(1)
        for k in range(len(centers)):
            m = labels == k
            ax.scatter(X[m, 0], X[m, 1], s=22, color=PALETTE[k % 10], alpha=0.75, edgecolor="white", linewidth=0.4)
            trail = hist[:frame + 1, k]
            ax.plot(trail[:, 0], trail[:, 1], "-", color="k", lw=1, alpha=0.5)
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, color="black", edgecolor="white", linewidth=1.2, zorder=5)
        ax.set_title(f"{title}\niteration {frame}/{len(hist) - 1}", fontsize=10)
    anim = FuncAnimation(fig, draw, frames=len(hist), interval=1000 // fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_optimizer_paths(fun, grad, x0, optimizers, xlim=(-2, 2), ylim=(-1, 3), n_iters=300,
                         ax=None, title="Optimizer race", save=None, levels=None):
    """Draw the trajectories of several optimizers on a 2-D loss surface.

    ``optimizers`` maps a label to ``(optimizer_name_or_instance, learning_rate)``.
    """
    from ._common import finish, prepare_axes
    ax = prepare_axes(ax, (7, 5.5))
    xs, ys = np.linspace(*xlim, 300), np.linspace(*ylim, 300)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array([[fun([a, b]) for a in xs] for b in ys])
    ax.contour(XX, YY, ZZ, levels=levels or np.logspace(-1, 3.8, 26), colors="#7a7a7a", alpha=0.6, linewidths=0.8)
    ax.grid(False)
    paths = {}
    for k, (name, (opt, lr)) in enumerate(optimizers.items()):
        p = minimize(fun, grad, x0, opt, n_iters, lr)
        paths[name] = p
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.8, color=PALETTE[k % 10], label=f"{name} (f = {fun(p[-1]):.3g})")
        ax.scatter(*p[-1], color=PALETTE[k % 10], s=36, zorder=5)
    ax.scatter(*x0, color="k", marker="o", s=45, label="start", zorder=6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    finish(ax, title, "x", "y", save, legend=True)
    return ax, paths


def animate_optimizer_paths(fun, grad, x0, optimizers, path="optimizers.gif", xlim=(-2, 2),
                            ylim=(-1, 3), n_iters=300, step=6, fps=15, levels=None):
    """Animated version of :func:`plot_optimizer_paths`."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt = get_plt()
    xs, ys = np.linspace(*xlim, 250), np.linspace(*ylim, 250)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.array([[fun([a, b]) for a in xs] for b in ys])
    paths = {n: minimize(fun, grad, x0, o, n_iters, lr) for n, (o, lr) in optimizers.items()}
    longest = max(len(p) for p in paths.values())
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.contour(XX, YY, ZZ, levels=levels or np.logspace(-1, 3.8, 26), colors="#7a7a7a", alpha=0.6, linewidths=0.8)
    ax.grid(False)
    lines, dots = {}, {}
    for k, name in enumerate(paths):
        lines[name], = ax.plot([], [], lw=1.8, color=PALETTE[k % 10], label=name)
        dots[name], = ax.plot([], [], "o", color=PALETTE[k % 10], ms=6)
    ax.scatter(*x0, color="k", s=40, zorder=6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    def update(f):
        i = min(f * step, longest - 1)
        for name, p in paths.items():
            j = min(i, len(p) - 1)
            lines[name].set_data(p[:j + 1, 0], p[:j + 1, 1])
            dots[name].set_data([p[j, 0]], [p[j, 1]])
        ax.set_title(f"Optimizer race - step {i}", fontsize=10)
        return list(lines.values()) + list(dots.values())
    anim = FuncAnimation(fig, update, frames=longest // step + 1, interval=1000 // fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
