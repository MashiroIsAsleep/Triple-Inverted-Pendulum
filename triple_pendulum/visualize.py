from __future__ import annotations

import numpy as np

from .model import TriplePendulumModel
from .simulation import SimulationResult


def _set_figure_window_size(fig, fraction: float = 0.9) -> None:
    manager = getattr(fig.canvas, "manager", None)
    window = getattr(manager, "window", None)
    if window is None:
        return

    try:
        if hasattr(window, "winfo_screenwidth") and hasattr(window, "winfo_screenheight"):
            screen_width = int(window.winfo_screenwidth())
            screen_height = int(window.winfo_screenheight())
            width = int(screen_width * fraction)
            height = int(screen_height * fraction)
            left = max(0, (screen_width - width) // 2)
            top = max(0, (screen_height - height) // 2)
            window.geometry(f"{width}x{height}+{left}+{top}")
            return

        if hasattr(window, "screen") and callable(window.screen):
            screen = window.screen()
            if screen is not None and hasattr(screen, "availableGeometry"):
                geometry = screen.availableGeometry()
                width = int(geometry.width() * fraction)
                height = int(geometry.height() * fraction)
                left = int(geometry.x() + (geometry.width() - width) / 2)
                top = int(geometry.y() + (geometry.height() - height) / 2)
                if hasattr(window, "resize"):
                    window.resize(width, height)
                if hasattr(window, "move"):
                    window.move(left, top)
    except Exception:
        pass


def plot_result(result: SimulationResult, model: TriplePendulumModel) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(result.time, result.states[:, 3])
    axes[0].set_ylabel("cart s [m]")
    axes[0].grid(True)

    for i in range(3):
        axes[1].plot(result.time, np.rad2deg(result.states[:, i]), label=f"phi{i + 1}")
    axes[1].set_ylabel("angle [deg]")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    label = "cart acceleration [m/s^2]" if result.mode == "acceleration" else "cart force [N]"
    axes[2].plot(result.time, result.controls)
    axes[2].set_ylabel(label)
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True)
    fig.tight_layout()
    _set_figure_window_size(fig)
    plt.show()


def animate_result(
    result: SimulationResult,
    model: TriplePendulumModel,
    stride: int = 8,
    follow_cart: bool = True,
    show: bool = True,
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if stride <= 0:
        raise ValueError("stride must be positive")

    states = result.states[::stride]
    times = result.time[::stride]
    controls = result.controls[::stride]
    link_total = float(np.sum(model.params.lengths))
    x_window = 3.0 * link_total
    link_colors = ("#2563eb", "#f59e0b", "#16a34a")
    joint_colors = ("#1d4ed8", "#d97706", "#15803d")
    depth_offsets = link_total * np.array([-0.18, 0.0, 0.18])
    depth_pad = 0.14 * link_total

    initial_cart = float(states[0, 3])
    if follow_cart:
        x_min = initial_cart - 0.5 * x_window
        x_max = initial_cart + 0.5 * x_window
    else:
        x_min = float(np.min(states[:, 3]) - 0.75 * link_total)
        x_max = float(np.max(states[:, 3]) + 0.75 * link_total)

    y_min = -1.15 * link_total
    y_max = 1.15 * link_total

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True)

    rail, = ax.plot([x_min, x_max], [0.0, 0.0], color="0.25", linewidth=2.0)
    cart, = ax.plot([], [], "s", color="#1f77b4", markersize=16)
    link_lines = [
        ax.plot([], [], "o-", color=color, linewidth=3.0, markersize=6)[0]
        for color in link_colors
    ]
    joints = [
        ax.plot([], [], "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=0.8)[0]
        for color in joint_colors
    ]
    tip, = ax.plot([], [], "o", color="#dc2626", markersize=7)
    label = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.8, "boxstyle": "round,pad=0.25"},
    )

    side = ax.inset_axes([0.70, 0.05, 0.26, 0.34])
    side.set_title("side view", fontsize=8)
    side.set_xlim(float(depth_offsets[0] - depth_pad), float(depth_offsets[-1] + depth_pad))
    side.set_ylim(y_min, y_max)
    side.set_xticks([])
    side.set_yticks([])
    side.grid(True, linewidth=0.4, alpha=0.35)
    side.axhline(0.0, color="0.35", linewidth=1.2)
    side.plot(depth_offsets, np.zeros(3), "s", color="#64748b", markersize=4)
    side_links = [
        side.plot([], [], "o-", color=color, linewidth=3.0, markersize=4)[0]
        for color in link_colors
    ]
    side_tip, = side.plot([], [], "o", color="#dc2626", markersize=5)

    def update(frame: int):
        state = states[frame]
        points = model.link_points(state)
        if follow_cart:
            center = float(state[3])
            left = center - 0.5 * x_window
            right = center + 0.5 * x_window
            ax.set_xlim(left, right)
            rail.set_data([left, right], [0.0, 0.0])
        cart.set_data([points[0, 0]], [points[0, 1]])
        for i, line in enumerate(link_lines):
            line.set_data(points[i : i + 2, 0], points[i : i + 2, 1])
            joints[i].set_data([points[i + 1, 0]], [points[i + 1, 1]])
        tip.set_data([points[-1, 0]], [points[-1, 1]])
        for i, line in enumerate(side_links):
            z = depth_offsets[i]
            line.set_data([z, z], [points[i, 1], points[i + 1, 1]])
        side_tip.set_data([depth_offsets[-1]], [points[-1, 1]])
        label.set_text(f"t = {times[frame]:.2f} s\nu = {controls[frame]:.2f}")
        return (
            rail,
            cart,
            *link_lines,
            *joints,
            tip,
            label,
            *side_links,
            side_tip,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=25,
        blit=False,
        repeat=True,
    )
    if save_path:
        animation.save(save_path)
    if show:
        _set_figure_window_size(fig)
        plt.show()
    return animation
