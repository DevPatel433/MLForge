"""Draw a fitted decision tree with matplotlib patches."""
import numpy as np

from ._common import PALETTE, finish, prepare_axes


def plot_tree(tree, feature_names=None, class_names=None, max_depth=None, ax=None,
              title="Learned decision tree", save=None, fontsize=7):
    """Render the splits, sample counts and (for classifiers) class colours."""
    from matplotlib.patches import FancyBboxPatch
    root = tree.root_
    names = feature_names or [f"x[{i}]" for i in range(tree.n_features_in_)]
    is_clf = hasattr(tree, "classes_")
    classes = class_names or ([str(c) for c in tree.classes_] if is_clf else None)

    # layout: x = in-order leaf position, y = -depth
    pos = {}
    counter = [0]

    def layout(nd, depth):
        if nd.is_leaf or (max_depth is not None and depth >= max_depth):
            pos[id(nd)] = (counter[0], -depth)
            counter[0] += 1
            return
        layout(nd.left, depth + 1)
        layout(nd.right, depth + 1)
        pos[id(nd)] = ((pos[id(nd.left)][0] + pos[id(nd.right)][0]) / 2, -depth)
    layout(root, 0)
    max_d = -min(p[1] for p in pos.values())
    width = max(6, counter[0] * 1.8)
    ax = prepare_axes(ax, (width, 1.4 * (max_d + 1) + 1))

    def draw(nd, depth):
        x, y = pos[id(nd)]
        leaf = nd.is_leaf or (max_depth is not None and depth >= max_depth)
        if not leaf:
            for child in (nd.left, nd.right):
                cx, cy = pos[id(child)]
                ax.plot([x, cx], [y - 0.3, cy + 0.3], color="#666", lw=1, zorder=1)
            ax.text((x + pos[id(nd.left)][0]) / 2 - 0.05, (y + pos[id(nd.left)][1]) / 2 + 0.05, "yes", fontsize=6, color="#555")
            ax.text((x + pos[id(nd.right)][0]) / 2 + 0.05, (y + pos[id(nd.right)][1]) / 2 + 0.05, "no", fontsize=6, color="#555")
            draw(nd.left, depth + 1)
            draw(nd.right, depth + 1)
        if is_clf:
            k = int(np.argmax(nd.value))
            color = PALETTE[k % 10]
            purity = nd.value[k] / nd.value.sum()
            label = f"{names[nd.feature]} <= {nd.threshold:.2f}\n" if not leaf else ""
            label += f"n = {nd.n_samples}\n{classes[k]} ({purity:.0%})"
            alpha = 0.25 + 0.5 * purity
        else:
            color, alpha = PALETTE[0], 0.35
            label = f"{names[nd.feature]} <= {nd.threshold:.2f}\n" if not leaf else ""
            label += f"n = {nd.n_samples}\nvalue = {nd.value:.2f}"
        box = FancyBboxPatch((x - 0.46, y - 0.3), 0.92, 0.6, boxstyle="round,pad=0.02",
                             fc=color, ec="#333", alpha=alpha, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=3)
    draw(root, 0)
    ax.set_xlim(-0.8, counter[0] - 0.2)
    ax.set_ylim(-max_d - 0.7, 0.7)
    ax.axis("off")
    return finish(ax, title, save=save)
