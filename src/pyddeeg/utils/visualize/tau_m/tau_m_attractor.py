# viz_surfaces.py  –  prettier 3-panel visualisation
# ---------------------------------------------------
from __future__ import annotations
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from nolitsa import delay as ndelay


def reconstruct(window: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    Takens embedding (N, m).
    """
    return ndelay.utils.reconstruct(window, dim=m, tau=tau)


def _surface_with_floor(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cmap: str,
    zlabel: str,
    mins_x: np.ndarray,
    mins_y: np.ndarray,
    subplot_label: str,
    title: str,
    elev: float,
    azim: float,
):
    """
    Draw a coloured 3D surface, project a lowered heat‐map beneath z=0,
    highlight minima on the surface, and add a subplot letter.
    """
    # 1) main 3D surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)

    # 2) draw a light z=0 reference plane
    Z0 = np.zeros_like(Z)
    # ax.plot_surface(X, Y, Z0, color='gray', alpha=0.2, linewidth=0)

    # 3) project the heatmap deeper, just below the data's minimum
    data_span = Z.max() - Z.min()
    z_floor = Z.min() - data_span * 0.05
    # ax.contourf(X, Y, Z, zdir='z', offset=z_floor, cmap=cmap, levels=60)

    # 4) highlight minima on the *surface* itself
    z_mins = Z[mins_x, mins_y]
    # ax.scatter(mins_x, mins_y, z_mins, c='red', s=30, marker='o', depthshade=True, label='minimum')

    # 5) aesthetics & view
    ax.set_zlim(z_floor, Z.max())
    ax.set(xlabel="Window Index", zlabel=zlabel, title=title)
    ax.view_init(elev=elev, azim=azim)

    # 6) subplot letter in top-left (axes coordinates)
    ax.text2D(
        0.02,
        0.93,
        subplot_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="normal",
    )

    # 7) colourbar for this surface
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(Z)


def _set_axes_equal(ax):
    """
    Make 3D axes have equal scale so that units are represented uniformly.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    center = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    new_limits = np.array([center - radius, center + radius]).T
    ax.set_xlim3d(new_limits[0])
    ax.set_ylim3d(new_limits[1])
    ax.set_zlim3d(new_limits[2])


def plot_tau_fnn_attractor(
    tau_results: List[Dict],
    m_results: List[Dict],
    windows: np.ndarray,
    win_to_show: int | None = None,
    *,
    elev: int = 25,
    azim: int = -55,
) -> None:
    """
    3-panel figure: MI surface, FNN surface, 3-D attractor for one window.
    """
    # pick the middle window by default
    n_win = len(tau_results)
    if win_to_show is None:
        win_to_show = n_win // 2 + 5

    # stack results into matrices
    mi_mat = np.stack([r["mi"] for r in tau_results])
    fnn_mat = np.stack([r["f3"] for r in m_results])

    W_tau, TAU = np.meshgrid(np.arange(n_win), tau_results[0]["taus"], indexing="ij")
    W_m, M = np.meshgrid(np.arange(n_win), m_results[0]["dims"], indexing="ij")

    # compute one trajectory
    tau_star = tau_results[win_to_show]["best_tau"]
    m_star = m_results[win_to_show]["best_m"]
    attract = reconstruct(windows[win_to_show], tau_star, m_star)

    # prepare figure
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("white")  # white around the panels

    # a) MI surface
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    mins_tau = np.array([r["best_tau"] for r in tau_results])
    _surface_with_floor(
        ax1,
        W_tau,
        TAU,
        mi_mat,
        cmap="viridis",
        zlabel="MI",
        mins_x=np.arange(n_win),
        mins_y=mins_tau - 1,
        subplot_label="a.",
        title="Mutial Information (MI) Surface",
        elev=elev,
        azim=azim,
    )
    ax1.set_ylabel(r"$\tau$")

    # b) FNN surface
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    mins_m = np.array([r["best_m"] for r in m_results])
    _surface_with_floor(
        ax2,
        W_m,
        M,
        fnn_mat,
        cmap="plasma",
        zlabel="FNN fraction",
        mins_x=np.arange(n_win),
        mins_y=mins_m - 1,
        subplot_label="b.",
        title="False Nearest Neighbours (FNN) Surface",
        elev=elev,
        azim=azim,
    )
    ax2.set_ylabel("m")

    # c) phase-space attractor
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot(attract[:, 0], attract[:, 1], attract[:, 2], lw=0.6, color="black")
    ax3.set(
        title=(
            f"Taken's Phase-space (window {win_to_show}, "
            f"$\\tau$={tau_star}, m={m_star})"
        ),
        xlabel="x(t)",
        ylabel=rf"x(t+{tau_star})",
        zlabel=rf"x(t+{2*tau_star})",
    )
    ax3.view_init(elev=elev, azim=azim)
    ax3.text2D(
        0.02, 0.93, "c.", transform=ax3.transAxes, fontsize=12, fontweight="normal"
    )

    # orthographic + equal aspect + zoom
    for ax in (ax1, ax2, ax3):
        ax.set_proj_type("ortho")
        ax.dist = 9

    # spread the panels out
    fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.08, wspace=0.05)
    # save with extra white padding so nothing is clipped
    plt.savefig(
        "tau_fnn_attractor.png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.2,
    )

    plt.show()
