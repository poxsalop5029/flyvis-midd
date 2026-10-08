"""Plotting helpers for rendered luminance and decoded flow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def show_renderer_comparison(result, title):
    """Show source, official rendering, manual rendering, and difference."""
    from flyvis.analysis.visualization import plots

    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    axes[0].imshow(result["image"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("input")
    axes[0].axis("off")
    for ax, key, label in zip(
        axes[1:],
        ("official", "manual", "difference"),
        ("official BoxEye", "manual crop mean", "difference"),
    ):
        values = result[key]
        limit = None if key != "difference" else max(values.abs().max().item(), 1e-7)
        plots.quick_hex_scatter(
            values,
            fig=fig,
            ax=ax,
            vmin=0 if key != "difference" else -limit,
            vmax=1 if key != "difference" else limit,
            cmap="gray" if key != "difference" else "coolwarm",
            cbar=False,
            fontsize=5,
        )
        ax.set_title(label)
    fig.suptitle(title)
    plt.tight_layout()
    return fig, axes


def plot_layout_rendering(ax, result, values, title, draw_outer_frame=True):
    """Plot receptor values and optional layout/image boundaries."""
    from matplotlib.patches import Rectangle

    centers = result["receptor_centers"]
    x = centers[..., 1].reshape(-1)
    y = centers[..., 0].reshape(-1)
    ax.scatter(
        x, y, c=values.reshape(-1), cmap="gray", vmin=0, vmax=1,
        marker="h", s=9, linewidths=0,
    )
    for field_id, (cy, cx) in enumerate(result["field_centers"]):
        ax.text(cx, cy, str(field_id), color="tab:red", ha="center", va="center")
    if draw_outer_frame:
        x0, x1 = float(x.min()), float(x.max())
        y0, y1 = float(y.min()), float(y.max())
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="tab:red", linewidth=1.5,
            label="layout outer frame",
        ))
        image_y, image_x = result["image_origin"].tolist()
        height, width = result["image"].shape
        ax.add_patch(Rectangle(
            (image_x, image_y), width, height,
            fill=False, edgecolor="tab:blue", linestyle="--", linewidth=1.2,
            label="placed image",
        ))
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return ax


def draw_hex_flow(
    ax,
    yx,
    values,
    flow_uv,
    title,
    color="red",
    stride=8,
    size=12,
    scale=None,
    width=0.0035,
    axis_off=False,
):
    """Draw rendered hexels and display-coordinate flow on one axis."""
    yx = torch.as_tensor(yx).detach().cpu().reshape(-1, 2)
    values = torch.as_tensor(values).detach().cpu().reshape(-1)
    flow_uv = torch.as_tensor(flow_uv).detach().cpu().reshape(-1, 2)
    if not (len(yx) == len(values) == len(flow_uv)):
        raise ValueError("yx, values, and flow_uv must have matching positions")
    index = torch.arange(0, len(yx), stride)
    ax.scatter(
        yx[:, 1], yx[:, 0], c=values, cmap="gray", vmin=0, vmax=1,
        marker="h", s=size, linewidths=0,
    )
    ax.quiver(
        yx[index, 1], yx[index, 0],
        flow_uv[index, 0], -flow_uv[index, 1],
        color=color, angles="xy", scale_units="xy", scale=scale, width=width,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if axis_off:
        ax.set_axis_off()
    return ax
