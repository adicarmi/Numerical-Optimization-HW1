
"""Utility helpers for visualising optimisation runs."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # head‑less backend for automated tests
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------

def plot_contours(func, xlim=None, ylim=None, ax=None, levels=20, **kwargs):
    """Plot contour lines and iterates for a 2-D objective, zooming on the iterates."""

    paths  = kwargs.pop('paths', None)
    labels = kwargs.pop('labels', None)
    title  = kwargs.pop('title',  None)
    save   = kwargs.pop('save',  None)

    # Stack all iterates for dynamic axis limits
    if paths is not None:
        all_pts = np.vstack(paths)
        x_min, x_max = all_pts[:,0].min(), all_pts[:,0].max()
        y_min, y_max = all_pts[:,1].min(), all_pts[:,1].max()
        x_margin = 0.5 * (x_max - x_min) if (x_max > x_min) else 1
        y_margin = 0.5 * (y_max - y_min) if (y_max > y_min) else 1
        xs = np.linspace(x_min - x_margin, x_max + x_margin, 400)
        ys = np.linspace(y_min - y_margin, y_max + y_margin, 400)
    else:
        xs = np.linspace(*(xlim or (-2,2)), 400)
        ys = np.linspace(*(ylim or (-2,2)), 400)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j], _, _ = func(np.array([X[i, j], Y[i, j]]), hess=False)

    # Contour levels around iterates
    path_fvals = []
    if paths is not None:
        for path in paths:
            for pt in path:
                fval, _, _ = func(np.asarray(pt), hess=False)
                path_fvals.append(fval)
    path_fvals = np.array(path_fvals) if path_fvals else Z.flatten()
    vmin = np.min(path_fvals) - 0.1 * abs(np.min(path_fvals))
    vmax = np.max(path_fvals) + 0.2 * abs(np.max(path_fvals))
    if vmax <= vmin:
        vmin, vmax = np.min(Z), np.max(Z)
    levels_auto = np.linspace(vmin, vmax, levels)

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Filled contours (background)
    cf = ax.contourf(X, Y, Z, levels=levels_auto, cmap='YlGnBu', alpha=0.65)
    # Line contours (for reference)
    cl = ax.contour(X, Y, Z, levels=levels_auto, colors='k', linewidths=0.8, alpha=0.7)
    if levels <= 15:
        ax.clabel(cl, inline=True, fontsize=8, fmt="%.1f")

    # Plot optimization paths
    if paths is not None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for k, path in enumerate(paths):
            lbl = labels[k] if labels else None
            ax.plot(path[:, 0], path[:, 1],
                    marker='o', markersize=8, linestyle='--', lw=2,
                    color=colors[k % len(colors)], label=lbl, alpha=0.9)
            
            #  annotate only start and end
            ax.annotate('start', path[0], textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8, color=colors[k % len(colors)])
            ax.annotate('end', path[-1], textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8, color=colors[k % len(colors)])
    # Set limits around iterates, with margin
    if paths is not None:
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
    else:
        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    if labels:
        ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, which='both', ls=':', alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=140, bbox_inches='tight')
        plt.close(ax.figure)
    return ax


# ---------------------------------------------------------------------
def plot_fvals(iter_fvals, labels, ax=None, **kwargs):
    """Plot objective value vs iteration for multiple methods."""
    save = kwargs.pop('save', None)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for fvals, lbl in zip(iter_fvals, labels):
        ax.plot(range(len(fvals)), fvals, marker='o', ms=3, lw=1.2, label=lbl)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective value')
    ax.grid(True, which='both', ls=':')
    ax.legend()
    if save:
        plt.savefig(save, dpi=200, bbox_inches='tight')
        plt.close(ax.figure)
    return ax
