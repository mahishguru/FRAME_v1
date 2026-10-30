#!/usr/bin/env python3
"""Plot-only script for the multi-run LoRA weight manifold.

Loads precomputed data from multi_run_manifold.py and generates paper figures.
Fast to iterate on — no kernel computation, no safetensor loading.

Usage:
    python plot_manifold.py                         # defaults
    python plot_manifold.py --elev 25 --azim -60    # tweak camera
    python plot_manifold.py --surface-alpha 0.5      # more transparent surface
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import proj3d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline, PchipInterpolator, RBFInterpolator
from scipy.spatial import Delaunay, QhullError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QWEN_DATA_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/manifold_analysis/multi_run_manifold_results"
)
QWEN_OUTPUT_DIR = QWEN_DATA_DIR
QWEN_PAPER_PLOTS_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/plots_paper/qwen3vl_manifold"
)
PIXTRAL_DATA_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/manifold_analysis/pixtral_multi_run_manifold_results"
)
PIXTRAL_OUTPUT_DIR = PIXTRAL_DATA_DIR
PIXTRAL_PAPER_PLOTS_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/plots_paper/pixtral_manifold"
)

PRESETS = {
    "qwen": {
        "data_dir": QWEN_DATA_DIR,
        "output_dir": QWEN_OUTPUT_DIR,
        "paper_plots_dir": QWEN_PAPER_PLOTS_DIR,
        "output_prefix": "qwen3vl",
    },
    "pixtral": {
        "data_dir": PIXTRAL_DATA_DIR,
        "output_dir": PIXTRAL_OUTPUT_DIR,
        "paper_plots_dir": PIXTRAL_PAPER_PLOTS_DIR,
        "output_prefix": "pixtral",
    },
}

DEFAULT_DATA_DIR = QWEN_DATA_DIR
DEFAULT_OUTPUT_DIR = QWEN_OUTPUT_DIR
DEFAULT_PAPER_PLOTS_DIR = QWEN_PAPER_PLOTS_DIR

HIGHLIGHT_RUN_NAME = "100% (set2)"
HIGHLIGHT_LABEL = "100% training trajectory"
HIGHLIGHT_COLOR = "#E41A1C"
BASE_COLOR = "#E69F00"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CheckpointPoint:
    run_name: str
    checkpoint_name: str
    step: Optional[int]
    epoch: Optional[float]
    loss: Optional[float]
    learning_rate: Optional[float]
    lora_scale: float


# ---------------------------------------------------------------------------
# Load precomputed data
# ---------------------------------------------------------------------------
def load_data(data_dir: Path):
    base_coord = np.load(data_dir / "multi_run_base_coord.npy")
    adapter_coords = np.load(data_dir / "multi_run_adapter_coords.npy")
    adapter_norms = np.load(data_dir / "multi_run_adapter_norms.npy")
    explained = np.load(data_dir / "multi_run_explained.npy")
    with (data_dir / "multi_run_points.json").open() as f:
        points_raw = json.load(f)
    all_points = [
        CheckpointPoint(
            run_name=p["run_name"],
            checkpoint_name=p["checkpoint_name"],
            step=p.get("step"),
            epoch=p.get("epoch"),
            loss=p.get("loss"),
            learning_rate=p.get("learning_rate"),
            lora_scale=p.get("lora_scale", 2.0),
        )
        for p in points_raw
    ]
    return all_points, adapter_coords, adapter_norms, base_coord, explained


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_fig(fig, stem, output_dir, paper_dir, formats):
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)
        fig.savefig(paper_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)


def checkpoint_label(point: CheckpointPoint) -> str:
    return str(point.step) if point.step is not None else "final"


def merge_checkpoint_labels(left: str, right: str) -> str:
    if left == right:
        return left
    if right == "final":
        return f"{left}/final"
    if left == "final":
        return f"{right}/final"
    return f"{left},{right}"


def highlighted_indices(points: Sequence[CheckpointPoint], require_loss: bool = False) -> List[int]:
    idx = [
        i for i, p in enumerate(points)
        if p.run_name == HIGHLIGHT_RUN_NAME and (not require_loss or p.loss is not None)
    ]
    if not idx:
        raise ValueError(f"No checkpoints found for highlighted run: {HIGHLIGHT_RUN_NAME}")
    return idx


def collapse_duplicate_trajectory_points(
    points: Sequence[CheckpointPoint],
    coords_2d: np.ndarray,
    values: np.ndarray,
    idx: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xy: List[np.ndarray] = []
    z: List[float] = []
    labels: List[str] = []
    for i in idx:
        label = checkpoint_label(points[i])
        value = float(values[i])
        if xy and np.linalg.norm(coords_2d[i] - xy[-1]) < 1e-8 and abs(value - z[-1]) < 1e-8:
            labels[-1] = merge_checkpoint_labels(labels[-1], label)
            continue
        xy.append(coords_2d[i].copy())
        z.append(value)
        labels.append(label)
    return np.vstack(xy), np.array(z, dtype=np.float64), labels


def padded_xy_limits(coords, pad_frac=0.16, min_pad=0.75):
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    xpad = max((xmax - xmin) * pad_frac, min_pad)
    ypad = max((ymax - ymin) * pad_frac, min_pad)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def build_rbf_surface(coords_2d, values, grid_n=180, pad_frac=0.08,
                      value_clip=None, mask_to_hull=True, smoothing=0.0):
    xmin, xmax = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    ymin, ymax = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    xpad = (xmax - xmin) * pad_frac
    ypad = (ymax - ymin) * pad_frac
    xi = np.linspace(xmin - xpad, xmax + xpad, grid_n)
    yi = np.linspace(ymin - ypad, ymax + ypad, grid_n)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_pts = np.column_stack([Xi.ravel(), Yi.ravel()])
    rbf = RBFInterpolator(coords_2d, values, kernel="thin_plate_spline", smoothing=smoothing)
    Zi = rbf(grid_pts).reshape(Xi.shape)
    if value_clip is not None:
        lo, hi = value_clip
        Zi = np.clip(Zi, -np.inf if lo is None else lo, np.inf if hi is None else hi)
    if mask_to_hull and len(coords_2d) >= 3:
        try:
            hull = Delaunay(coords_2d)
            outside = hull.find_simplex(grid_pts) < 0
            flat = Zi.ravel()
            flat[outside] = np.nan
            Zi = flat.reshape(Xi.shape)
        except QhullError:
            pass
    return Xi, Yi, Zi


def smooth_surface_trajectory(xy, surface_rbf, z_clip, points_per_segment=90, waypoint_z=None):
    if len(xy) < 2:
        z = np.asarray(surface_rbf(xy), dtype=np.float64)
        return xy.copy(), np.clip(z, z_clip[0], z_clip[1])

    chord = np.zeros(len(xy), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    chord[1:] = np.cumsum(segment_lengths)
    if len(xy) >= 3 and chord[-1] > 1e-12:
        tangents = np.zeros_like(xy, dtype=np.float64)
        for idx in range(len(xy)):
            if idx == 0:
                lookahead = 2 if len(xy) > 2 else 1
                denom = max(chord[lookahead] - chord[0], 1e-12)
                tangents[idx] = (xy[lookahead] - xy[0]) / denom
            elif idx == len(xy) - 1:
                lookback = len(xy) - 3 if len(xy) > 2 else len(xy) - 2
                denom = max(chord[-1] - chord[lookback], 1e-12)
                tangents[idx] = (xy[-1] - xy[lookback]) / denom
            else:
                denom = max(chord[idx + 1] - chord[idx - 1], 1e-12)
                tangents[idx] = (xy[idx + 1] - xy[idx - 1]) / denom

        parts = []
        path_s_parts = []
        for seg_idx in range(len(xy) - 1):
            local_t = np.linspace(0.0, 1.0, points_per_segment,
                                  endpoint=(seg_idx == len(xy) - 2))
            local_t2 = local_t * local_t
            local_t3 = local_t2 * local_t
            basis00 = 2 * local_t3 - 3 * local_t2 + 1
            basis10 = local_t3 - 2 * local_t2 + local_t
            basis01 = -2 * local_t3 + 3 * local_t2
            basis11 = local_t3 - local_t2
            segment_length = max(chord[seg_idx + 1] - chord[seg_idx], 1e-12)
            segment = (
                basis00[:, None] * xy[seg_idx]
                + basis10[:, None] * segment_length * tangents[seg_idx]
                + basis01[:, None] * xy[seg_idx + 1]
                + basis11[:, None] * segment_length * tangents[seg_idx + 1]
            )
            parts.append(segment)
            path_s_parts.append(chord[seg_idx] + local_t * segment_length)
        smooth_xy = np.vstack(parts)
        smooth_s = np.concatenate(path_s_parts)
    else:
        parts = []
        path_s_parts = []
        for seg_idx in range(len(xy) - 1):
            local_t = np.linspace(0, 1, points_per_segment, endpoint=(seg_idx == len(xy) - 2))
            segment_length = max(chord[seg_idx + 1] - chord[seg_idx], 1e-12)
            parts.append(np.column_stack([
                xy[seg_idx, 0] + local_t * (xy[seg_idx + 1, 0] - xy[seg_idx, 0]),
                xy[seg_idx, 1] + local_t * (xy[seg_idx + 1, 1] - xy[seg_idx, 1]),
            ]))
            path_s_parts.append(chord[seg_idx] + local_t * segment_length)
        smooth_xy = np.vstack(parts)
        smooth_s = np.concatenate(path_s_parts)

    if waypoint_z is not None and len(waypoint_z) == len(xy):
        z_interp = PchipInterpolator(chord, np.asarray(waypoint_z, dtype=np.float64), extrapolate=True)
        smooth_z = np.asarray(z_interp(smooth_s), dtype=np.float64)
    else:
        smooth_z = np.asarray(surface_rbf(smooth_xy), dtype=np.float64)
    smooth_z = np.clip(smooth_z, z_clip[0], z_clip[1])
    return smooth_xy, smooth_z


def project_3d_to_2d(ax, x, y, z):
    return proj3d.proj_transform(np.asarray(x), np.asarray(y), np.asarray(z), ax.get_proj())[:2]


def make_checkered_flag_box():
    flag = DrawingArea(48, 40, 0, 0)
    flag.add_artist(Line2D([8, 8], [2, 38], color="0.12", lw=1.6))
    cell = 7
    x0, y0 = 8, 28
    for row in range(3):
        for col in range(4):
            color = "black" if (row + col) % 2 == 0 else "white"
            flag.add_artist(Rectangle((x0 + col * cell, y0 - row * cell), cell, cell,
                                      facecolor=color, edgecolor="0.20", linewidth=0.35))
    return flag


def add_paper_note(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
            edgecolor="0.75", alpha=0.86), zorder=10)


def annotate_2d_steps(ax, xy, labels, color="black"):
    offsets = [(0, -16), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10)]
    for k, (point, label) in enumerate(zip(xy, labels)):
        ann = ax.annotate(label, (point[0], point[1]),
                          xytext=offsets[min(k, len(offsets) - 1)],
                          textcoords="offset points", ha="center", va="center",
                          fontsize=8.5, color=color,
                          bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                    edgecolor="none", alpha=0.72), zorder=8)
        ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def draw_highlight_trajectory_2d(ax, xy, labels, label=HIGHLIGHT_LABEL):
    ax.plot(xy[:, 0], xy[:, 1], "-", color="black", lw=5.2, alpha=0.45, zorder=5)
    ax.plot(xy[:, 0], xy[:, 1], "-o", color=HIGHLIGHT_COLOR, lw=3.4, ms=9.5,
            label=label, markerfacecolor=HIGHLIGHT_COLOR, markeredgecolor="white",
            markeredgewidth=1.3, zorder=6)
    for k in range(1, len(xy)):
        ax.annotate("", xy=(xy[k, 0], xy[k, 1]), xytext=(xy[k - 1, 0], xy[k - 1, 1]),
                     arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT_COLOR, lw=2.0,
                                     mutation_scale=16, shrinkA=15, shrinkB=15), zorder=7)
    annotate_2d_steps(ax, xy, labels)


# ---------------------------------------------------------------------------
# 3D Weight Manifold Plot
# ---------------------------------------------------------------------------
def plot_3d_manifold(
    all_points, adapter_coords, adapter_norms, base_coord, explained,
    output_dir, paper_dir, formats, output_prefix="qwen3vl",
    # ---- tuneable knobs ----
    elev=30, azim=-55, focal_length=0.82,
    surface_alpha=0.45, contour_alpha=0.70, floor_alpha=0.18,
    n_contour_levels=14, n_floor_levels=16,
    traj_lift_frac=0.14,
    line_width=4.2, outline_width=7.0,
    marker_size=78,
    box_aspect=(1.25, 1.22, 0.88),
    figsize=(9.5, 7.0),
    light_azimuth=315, light_altitude=40,
    pc1_min=-6.0,
    crop_to_trajectory=True,
    crop_margin=4.00,
    crop_margin_pc1_pos=5.00,
    crop_margin_pc2_neg=7.00,
    crop_margin_pc2_pos=7.00,
    xlim=(-5.0, 18.0),
    ylim=(-8.0, 8.0),
    z_headroom=1.58,
    pc3_max=20.0,
    surface_rbf_smoothing=0.85,
    surface_smooth_sigma=2.8,
    pad_frac=0.20, grid_n=200,
):
    # Build the RBF surface using ALL checkpoints + base
    surface_coords = np.vstack([base_coord[None, :], adapter_coords])
    surface_norms = np.concatenate([[0.0], adapter_norms])
    seen = {}
    unique_idx = []
    for i, c in enumerate(surface_coords):
        key = (round(c[0], 10), round(c[1], 10))
        if key not in seen:
            seen[key] = i
            unique_idx.append(i)
    c2_unique = surface_coords[unique_idx]
    norms_unique = surface_norms[unique_idx]

    Xi_full, Yi_full, Zi_full = build_rbf_surface(
        c2_unique, norms_unique, grid_n=grid_n, pad_frac=pad_frac,
        value_clip=(0.0, float(norms_unique.max()) * 1.10), mask_to_hull=False,
        smoothing=surface_rbf_smoothing,
    )

    idx = highlighted_indices(all_points)
    traj_xy, traj_z, traj_labels = collapse_duplicate_trajectory_points(
        all_points, adapter_coords, adapter_norms, idx)

    crop_xy = np.vstack([base_coord[None, :], traj_xy])
    crop_xlim = (max(pc1_min, float(crop_xy[:, 0].min()) - crop_margin),
                 float(crop_xy[:, 0].max()) + crop_margin_pc1_pos)
    crop_ylim = (float(crop_xy[:, 1].min()) - crop_margin_pc2_neg,
                 float(crop_xy[:, 1].max()) + crop_margin_pc2_pos)
    if xlim is not None:
        crop_xlim = tuple(xlim)
    if ylim is not None:
        crop_ylim = tuple(ylim)

    x_exp = explained[0] * 100
    y_exp = explained[1] * 100

    fig = plt.figure(figsize=figsize)
    ax3 = fig.add_subplot(111, projection="3d")

    zmax_3d = min(max(float(traj_z.max()) * z_headroom, 1.0), pc3_max)
    Zi_3d = np.clip(Zi_full, 0.0, zmax_3d)
    Xi_3d, Yi_3d = Xi_full, Yi_full
    if crop_to_trajectory:
        x_keep = (Xi_full[0, :] >= crop_xlim[0]) & (Xi_full[0, :] <= crop_xlim[1])
        y_keep = (Yi_full[:, 0] >= crop_ylim[0]) & (Yi_full[:, 0] <= crop_ylim[1])
        Xi_3d = Xi_full[np.ix_(y_keep, x_keep)]
        Yi_3d = Yi_full[np.ix_(y_keep, x_keep)]
        Zi_3d = Zi_3d[np.ix_(y_keep, x_keep)]
    Zi_plot = gaussian_filter(Zi_3d, sigma=surface_smooth_sigma, mode="nearest")

    # Lit surface
    light = LightSource(azdeg=light_azimuth, altdeg=light_altitude)
    facecolors = light.shade(Zi_plot, cmap=cm.cividis, vert_exag=0.6, blend_mode="soft")
    facecolors[..., 3] = surface_alpha
    ax3.plot_surface(Xi_3d, Yi_3d, Zi_plot, facecolors=facecolors, linewidth=0,
                     antialiased=True, rcount=160, ccount=160, shade=False, zorder=1)

    # Contour the smoothed plotting surface so sparse RBF wiggles do not create
    # broken-looking isolines in extrapolated regions.
    z_lo = float(np.nanpercentile(Zi_plot, 1))
    z_hi = float(np.nanpercentile(Zi_plot, 99))
    contour_levels = np.linspace(z_lo, z_hi, n_contour_levels)
    ax3.contour(Xi_3d, Yi_3d, Zi_plot, levels=contour_levels,
                colors="0.02", linewidths=0.64, alpha=contour_alpha, zorder=2)

    # Trajectory on the surface
    rbf_traj = RBFInterpolator(c2_unique, norms_unique,
                               kernel="thin_plate_spline", smoothing=0.0)
    full_traj_xy = np.vstack([base_coord[None, :], traj_xy])
    full_traj_z = np.concatenate([[0.0], traj_z])
    full_traj_labels = ["base"] + list(traj_labels)

    smooth_xy, smooth_z = smooth_surface_trajectory(
        full_traj_xy, rbf_traj, (0.0, zmax_3d), points_per_segment=100,
        waypoint_z=full_traj_z)
    rx, ry, rz = smooth_xy[:, 0], smooth_xy[:, 1], smooth_z
    rz_lift = rz + zmax_3d * traj_lift_frac

    wp_z = np.clip(full_traj_z, 0.0, zmax_3d)
    wp_z_lift = wp_z + zmax_3d * traj_lift_frac

    ax3.set_xlabel(f"PC1 ({x_exp:.1f}% var.)", labelpad=16, fontsize=17)
    ax3.set_ylabel(f"PC2 ({y_exp:.1f}% var.)", labelpad=16, fontsize=17)
    ax3.set_zlabel(r"$\|\Delta W\|_F$", labelpad=14, fontsize=17)
    if crop_to_trajectory:
        ax3.set_xlim(*crop_xlim)
        ax3.set_ylim(*crop_ylim)
    else:
        ax3.set_xlim(max(pc1_min, float(Xi_full.min())), float(Xi_full.max()))
        ax3.set_ylim(float(Yi_full.min()), float(Yi_full.max()))
    ax3.set_zlim(0.0, zmax_3d)
    ax3.set_xticks(np.arange(-5.0, 18.1, 5.0))
    ax3.set_yticks(np.arange(-8.0, 8.1, 2.0))
    ax3.set_zticks(np.arange(0.0, 20.1, 2.5))
    ax3.view_init(elev=elev, azim=azim)
    try:
        ax3.set_proj_type("persp", focal_length=focal_length)
    except TypeError:
        ax3.set_proj_type("persp")
    ax3.set_box_aspect(list(box_aspect))

    # Matplotlib's native 3D artists are depth-sorted against the surface, so
    # render the important trajectory as a projected 2D overlay after camera setup.
    line_x2, line_y2 = project_3d_to_2d(ax3, rx, ry, rz_lift)
    ax3.add_line(Line2D(line_x2, line_y2, transform=ax3.transData, color="black",
                       lw=outline_width + 1.2, alpha=0.82, solid_capstyle="round",
                       zorder=1000, clip_on=False))
    ax3.add_line(Line2D(line_x2, line_y2, transform=ax3.transData, color=HIGHLIGHT_COLOR,
                       lw=line_width + 0.8, alpha=1.0, solid_capstyle="round",
                       label=HIGHLIGHT_LABEL, zorder=1001, clip_on=False))

    point_x2, point_y2 = project_3d_to_2d(
        ax3, full_traj_xy[:, 0], full_traj_xy[:, 1], wp_z_lift)
    ax3.add_line(Line2D(point_x2[1:], point_y2[1:], transform=ax3.transData,
                       linestyle="None", marker="o", markersize=np.sqrt(marker_size),
                       markerfacecolor=HIGHLIGHT_COLOR, markeredgecolor="white",
                       markeredgewidth=1.4, zorder=1002, clip_on=False))
    ax3.add_line(Line2D([point_x2[0]], [point_y2[0]], transform=ax3.transData,
                       linestyle="None", marker="*", markersize=18,
                       markerfacecolor=BASE_COLOR, markeredgecolor="black",
                       markeredgewidth=1.1, zorder=1002, clip_on=False))

    skip_labels = {"348", "300"}
    label_offsets = [(0, -17), (0, 13), (0, 13), (0, 13), (0, 13), (0, 15), (0, 17)]
    for k, lbl in enumerate(full_traj_labels):
        if "final" in lbl or lbl in skip_labels:
            continue
        dx, dy = (0, -18) if lbl == "250" else label_offsets[min(k, len(label_offsets) - 1)]
        ann = ax3.annotate(lbl, xy=(point_x2[k], point_y2[k]), xycoords=ax3.transData,
                           xytext=(dx, dy), textcoords="offset points",
                   ha="center", va="center", fontsize=12, color="black",
                           weight="bold", zorder=1003, clip_on=False)
        ann.set_path_effects([pe.withStroke(linewidth=3.8, foreground="white")])

    final_k = len(full_traj_xy) - 1
    flag = AnnotationBbox(make_checkered_flag_box(), (point_x2[final_k], point_y2[final_k]),
                          xybox=(20, 26), xycoords=ax3.transData,
                          boxcoords="offset points", frameon=False, pad=0,
                          zorder=1004, annotation_clip=False)
    ax3.add_artist(flag)
    final_ann = ax3.annotate("final", xy=(point_x2[final_k], point_y2[final_k]),
                             xycoords=ax3.transData, xytext=(16, 6),
                             textcoords="offset points", ha="center", va="center",
                             fontsize=12, color="black", weight="bold",
                             zorder=1005, clip_on=False)
    final_ann.set_path_effects([pe.withStroke(linewidth=3.8, foreground="white")])

    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor((0, 0, 0, 0.28))
    ax3.yaxis.pane.set_edgecolor((0, 0, 0, 0.28))
    ax3.zaxis.pane.set_edgecolor((0, 0, 0, 0.28))
    ax3.grid(False)
    # Tune the 3D frame/grid through the axis internals; standard grid kwargs
    # are inconsistently honored by mplot3d.
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis._axinfo["grid"].update({
            "color": (0, 0, 0, 0.20),
            "linewidth": 0.60,
            "linestyle": "-",
        })
        axis._axinfo["axisline"].update({
            "color": (0, 0, 0, 0.92),
            "linewidth": 2.0,
        })
        axis.line.set_color("black")
        axis.line.set_linewidth(2.0)
        axis.set_pane_color((1, 1, 1, 0))
    ax3.grid(True)
    ax3.tick_params(labelsize=13, width=1.65, pad=4)
    fig.subplots_adjust(left=-0.06, right=1.00, bottom=-0.03, top=0.98)
    save_fig(fig, f"{output_prefix}_multi_run_weight_manifold_3d", output_dir, paper_dir, formats)
    plt.close(fig)
    print(f"  3D plot saved (elev={elev}, azim={azim})")


# ---------------------------------------------------------------------------
# 2D Contour Plot
# ---------------------------------------------------------------------------
def plot_2d_manifold(
    all_points, adapter_coords, adapter_norms, base_coord, explained,
    output_dir, paper_dir, formats, output_prefix="qwen3vl",
):
    surface_coords = np.vstack([base_coord[None, :], adapter_coords])
    surface_norms = np.concatenate([[0.0], adapter_norms])
    seen = {}
    unique_idx = []
    for i, c in enumerate(surface_coords):
        key = (round(c[0], 10), round(c[1], 10))
        if key not in seen:
            seen[key] = i
            unique_idx.append(i)
    c2_unique = surface_coords[unique_idx]
    norms_unique = surface_norms[unique_idx]

    Xi, Yi, Zi = build_rbf_surface(c2_unique, norms_unique, grid_n=220,
                                   value_clip=(0.0, float(norms_unique.max()) * 1.05),
                                   mask_to_hull=True)
    Zi_masked = np.ma.masked_invalid(Zi)

    idx = highlighted_indices(all_points)
    traj_xy, traj_z, traj_labels = collapse_duplicate_trajectory_points(
        all_points, adapter_coords, adapter_norms, idx)
    focus_xy = np.vstack([base_coord[None, :], traj_xy])
    xlim, ylim = padded_xy_limits(focus_xy, pad_frac=0.13, min_pad=0.85)
    x_exp = explained[0] * 100
    y_exp = explained[1] * 100

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    contour = ax.contourf(Xi, Yi, Zi_masked, levels=34, cmap="cividis", alpha=0.92)
    ax.contour(Xi, Yi, Zi_masked, levels=16, colors="black", linewidths=0.35, alpha=0.18)
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\|\Delta W\|_F$ (LoRA update norm)")

    full_traj_xy_2d = np.vstack([base_coord[None, :], traj_xy])
    full_traj_labels_2d = ["base"] + list(traj_labels)
    draw_highlight_trajectory_2d(ax, full_traj_xy_2d, full_traj_labels_2d)

    ax.scatter([base_coord[0]], [base_coord[1]], marker="*", s=360, color=BASE_COLOR,
               edgecolor="black", linewidth=1.2, zorder=9, label="base (pretrained)")

    ax.set_xlabel(f"PC1 ({x_exp:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({y_exp:.1f}% variance)")
    ax.set_title("100% Fine-Tuning Trajectory on LoRA Weight Manifold\n"
                 "(surface fitted from all 5 runs; only the 100% path is shown)")
    add_paper_note(ax, "Surface fit: 32 checkpoints from 5 runs\nPlotted trajectory: 100% fine-tuning run")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, f"{output_prefix}_multi_run_weight_manifold_2d", output_dir, paper_dir, formats)
    plt.close(fig)
    print("  2D plot saved")


# ---------------------------------------------------------------------------
# Loss landscape plots
# ---------------------------------------------------------------------------
def plot_loss_landscape(
    all_points, adapter_coords, base_coord, explained,
    output_dir, paper_dir, formats, output_prefix="qwen3vl",
):
    has_loss = [(i, p) for i, p in enumerate(all_points) if p.loss is not None]
    if len(has_loss) < 4:
        print("  Skipping loss landscape: fewer than 4 checkpoints have recorded loss.")
        return

    idx_loss = [i for i, _ in has_loss]
    c2_loss = adapter_coords[idx_loss]
    loss_vals = np.array([p.loss for _, p in has_loss], dtype=np.float64)

    seen = {}
    unique_idx = []
    for k in range(len(idx_loss)):
        key = (round(c2_loss[k, 0], 10), round(c2_loss[k, 1], 10))
        if key not in seen:
            seen[key] = k
            unique_idx.append(k)
    c2_u = c2_loss[unique_idx]
    loss_u = loss_vals[unique_idx]
    if len(c2_u) < 4:
        print("  Skipping loss landscape: fewer than 4 unique coordinate positions with loss.")
        return

    loss_pad = max(float(loss_vals.max() - loss_vals.min()) * 0.04, 1e-3)
    Xi, Yi, Zi = build_rbf_surface(c2_u, loss_u, grid_n=220,
        value_clip=(max(0.0, float(loss_vals.min()) - loss_pad), float(loss_vals.max()) + loss_pad))
    Zi_masked = np.ma.masked_invalid(Zi)

    hi_idx = highlighted_indices(all_points, require_loss=True)
    loss_by_point = np.array([np.nan if p.loss is None else p.loss for p in all_points], dtype=np.float64)
    traj_xy, traj_loss, traj_labels = collapse_duplicate_trajectory_points(
        all_points, adapter_coords, loss_by_point, hi_idx)
    focus_xy = np.vstack([base_coord[None, :], traj_xy])
    xlim, ylim = padded_xy_limits(focus_xy, pad_frac=0.13, min_pad=0.85)
    x_exp = explained[0] * 100
    y_exp = explained[1] * 100

    # --- 3D loss surface ---
    rbf_loss = RBFInterpolator(c2_u, loss_u, kernel="thin_plate_spline", smoothing=0.0)
    Xi_full_loss, Yi_full_loss, Zi_full_loss = build_rbf_surface(
        c2_u, loss_u, grid_n=200, pad_frac=0.20,
        value_clip=(max(0.0, float(loss_vals.min()) - loss_pad), float(loss_vals.max()) + loss_pad),
        mask_to_hull=False)

    base_loss_est = float(np.clip(rbf_loss(base_coord[None, :]), 0.0, None)[0])
    full_traj_xy = np.vstack([base_coord[None, :], traj_xy])
    full_traj_loss = np.concatenate([[base_loss_est], traj_loss])
    full_traj_labels_3d = ["base"] + list(traj_labels)

    smooth_xy, smooth_z = smooth_surface_trajectory(full_traj_xy, rbf_loss,
        (max(0.0, float(loss_vals.min()) - loss_pad), float(loss_vals.max()) + loss_pad))
    zfloor_loss = max(0.0, float(loss_vals.min()) - loss_pad)

    fig = plt.figure(figsize=(9.5, 7.2))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot_surface(Xi_full_loss, Yi_full_loss, Zi_full_loss, cmap="coolwarm_r",
                     alpha=0.42, linewidth=0, antialiased=True, zorder=1)
    ax3.contour(Xi_full_loss, Yi_full_loss, Zi_full_loss, levels=18,
                cmap="coolwarm_r", linewidths=0.7, alpha=0.6, zorder=2)
    ax3.contourf(Xi_full_loss, Yi_full_loss, Zi_full_loss, levels=20,
                 cmap="coolwarm_r", alpha=0.30, zdir="z", offset=zfloor_loss, zorder=0)

    rx, ry, rz = smooth_xy[:, 0], smooth_xy[:, 1], smooth_z
    ax3.plot(rx, ry, np.full_like(rz, zfloor_loss), "-", color="gray", lw=1.6, alpha=0.30, zorder=3)
    ax3.plot(rx, ry, rz, "-", color="black", lw=5.5, alpha=0.50, solid_capstyle="round", zorder=5)
    ax3.plot(rx, ry, rz, "-", color=HIGHLIGHT_COLOR, lw=3.4, label=HIGHLIGHT_LABEL, zorder=6)

    wp_z = rbf_loss(full_traj_xy)
    ax3.scatter(full_traj_xy[:, 0], full_traj_xy[:, 1], wp_z,
                color=HIGHLIGHT_COLOR, s=80, edgecolor="white", linewidth=1.4, zorder=7, depthshade=False)
    ax3.scatter([base_coord[0]], [base_coord[1]], [base_loss_est], marker="*", s=400,
                color=BASE_COLOR, edgecolor="black", linewidth=1.2, zorder=9,
                label="base (pretrained)", depthshade=False)

    ax3.set_xlabel(f"PC1 ({x_exp:.1f}%)", labelpad=10)
    ax3.set_ylabel(f"PC2 ({y_exp:.1f}%)", labelpad=10)
    ax3.set_zlabel("Training Loss", labelpad=10)
    ax3.set_title("100% Training Trajectory on LoRA Loss Surface")
    zpad = max(float(full_traj_loss.max() - full_traj_loss.min()) * 0.18, 0.03)
    ax3.set_zlim(max(0.0, float(full_traj_loss.min()) - zpad), float(full_traj_loss.max()) + zpad)
    ax3.view_init(elev=26, azim=-63)
    ax3.set_box_aspect([1.55, 0.75, 0.60])
    ax3.legend(fontsize=8.5, loc="upper left")
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.grid(True, alpha=0.08, linestyle=":", linewidth=0.5)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94)
    save_fig(fig, f"{output_prefix}_multi_run_loss_landscape_3d", output_dir, paper_dir, formats)
    plt.close(fig)
    print("  3D loss landscape saved")

    # --- 2D loss contour ---
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    contour = ax.contourf(Xi, Yi, Zi_masked, levels=34, cmap="coolwarm_r", alpha=0.88)
    ax.contour(Xi, Yi, Zi_masked, levels=16, colors="black", linewidths=0.35, alpha=0.18)
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Training Loss")
    full_traj_xy_2d = np.vstack([base_coord[None, :], traj_xy])
    full_traj_labels_2d = ["base"] + list(traj_labels)
    draw_highlight_trajectory_2d(ax, full_traj_xy_2d, full_traj_labels_2d)
    ax.scatter([base_coord[0]], [base_coord[1]], marker="*", s=360, color=BASE_COLOR,
               edgecolor="black", linewidth=1.2, zorder=9, label="base")
    ax.set_xlabel(f"PC1 ({x_exp:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({y_exp:.1f}% variance)")
    ax.set_title("100% Fine-Tuning Trajectory on LoRA Loss Surface")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, f"{output_prefix}_multi_run_loss_landscape_2d", output_dir, paper_dir, formats)
    plt.close(fig)
    print("  2D loss landscape saved")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot multi-run LoRA manifold from precomputed data (fast iteration)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="qwen")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--paper-plots-dir", type=Path, default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    # 3D camera knobs
    p.add_argument("--elev", type=float, default=30, help="Camera elevation angle")
    p.add_argument("--azim", type=float, default=-55, help="Camera azimuth angle")
    p.add_argument("--focal-length", type=float, default=0.82)
    # Surface appearance
    p.add_argument("--surface-alpha", type=float, default=0.45, help="Surface transparency 0-1")
    p.add_argument("--contour-alpha", type=float, default=0.70)
    p.add_argument("--floor-alpha", type=float, default=0.18)
    p.add_argument("--pc1-min", type=float, default=-6.0, help="Minimum PC1 value shown in the 3D plot")
    p.add_argument("--no-crop-to-trajectory", action="store_true", help="Show the full 3D fitted surface")
    p.add_argument("--crop-margin", type=float, default=4.00, help="PC margin around the highlighted trajectory for cropped 3D plots")
    p.add_argument("--crop-margin-pc1-pos", type=float, default=5.00, help="Extra margin after the highlighted trajectory on positive PC1")
    p.add_argument("--crop-margin-pc2-neg", type=float, default=7.00, help="Extra margin below the highlighted trajectory on negative PC2")
    p.add_argument("--crop-margin-pc2-pos", type=float, default=7.00, help="Extra margin above the highlighted trajectory on positive PC2")
    p.add_argument("--xlim", type=float, nargs=2, default=(-5.0, 18.0), metavar=("MIN", "MAX"), help="Fixed 3D PC1 axis limits")
    p.add_argument("--ylim", type=float, nargs=2, default=(-8.0, 8.0), metavar=("MIN", "MAX"), help="Fixed 3D PC2 axis limits")
    p.add_argument("--z-headroom", type=float, default=1.58, help="Multiplier for z-axis headroom above the trajectory")
    p.add_argument("--pc3-max", type=float, default=20.0, help="Maximum displayed PC3/z value for the 3D plot")
    p.add_argument("--surface-rbf-smoothing", type=float, default=0.85, help="RBF regularization for the displayed 3D surface/isolines")
    p.add_argument("--surface-smooth-sigma", type=float, default=2.8, help="Gaussian smoothing sigma for the displayed 3D surface/isolines")
    # Trajectory appearance
    p.add_argument("--traj-lift", type=float, default=0.14, help="Lift trajectory above surface (fraction of zmax)")
    p.add_argument("--line-width", type=float, default=4.2)
    p.add_argument("--outline-width", type=float, default=7.0)
    # Which plots
    p.add_argument("--only-3d", action="store_true", help="Only regenerate the 3D weight manifold plot")
    p.add_argument("--skip-loss", action="store_true", help="Skip loss landscape plots")
    return p.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    data_dir = (args.data_dir or preset["data_dir"]).resolve()
    output_dir = (args.output_dir or preset["output_dir"]).resolve()
    paper_dir = (args.paper_plots_dir or preset["paper_plots_dir"]).resolve()
    output_prefix = args.output_prefix or preset["output_prefix"]

    print(f"Loading precomputed data from {data_dir} ...")
    all_points, adapter_coords, adapter_norms, base_coord, explained = load_data(data_dir)
    print(f"  {len(all_points)} checkpoints, PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}%")

    if not args.only_3d:
        plot_2d_manifold(all_points, adapter_coords, adapter_norms, base_coord, explained,
                         output_dir, paper_dir, args.formats, output_prefix=output_prefix)

    plot_3d_manifold(
        all_points, adapter_coords, adapter_norms, base_coord, explained,
        output_dir, paper_dir, args.formats, output_prefix=output_prefix,
        elev=args.elev, azim=args.azim, focal_length=args.focal_length,
        surface_alpha=args.surface_alpha, contour_alpha=args.contour_alpha,
        floor_alpha=args.floor_alpha, traj_lift_frac=args.traj_lift,
        line_width=args.line_width, outline_width=args.outline_width,
        pc1_min=args.pc1_min,
        crop_to_trajectory=not args.no_crop_to_trajectory,
        crop_margin=args.crop_margin,
        crop_margin_pc1_pos=args.crop_margin_pc1_pos,
        crop_margin_pc2_neg=args.crop_margin_pc2_neg,
        crop_margin_pc2_pos=args.crop_margin_pc2_pos,
        xlim=args.xlim,
        ylim=args.ylim,
        z_headroom=args.z_headroom,
        pc3_max=args.pc3_max,
        surface_rbf_smoothing=args.surface_rbf_smoothing,
        surface_smooth_sigma=args.surface_smooth_sigma,
    )

    if not args.only_3d and not args.skip_loss:
        plot_loss_landscape(all_points, adapter_coords, base_coord, explained,
                            output_dir, paper_dir, args.formats, output_prefix=output_prefix)

    print(f"\nDone. Plots: {paper_dir}")


if __name__ == "__main__":
    main()
