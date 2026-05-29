#!/usr/bin/env python3
"""Multi-trajectory LoRA weight manifold.

Pools checkpoints from multiple fine-tuning runs (different data percentages)
into a single exact-kernel computation, yielding a genuine 2D manifold surface
from the weight-update geometry rather than from a single interpolated curve.

The smooth surface uses Radial Basis Function interpolation over the top-2
kernel-PCA coordinates with the LoRA update norm as the scalar field height.
All runs support the surface fit, but only the 100% fine-tuning trajectory is
drawn in the paper figures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from safetensors.torch import safe_open

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.patheffects as pe
from scipy.interpolate import CubicSpline, RBFInterpolator
from scipy.spatial import Delaunay, QhullError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"

QWEN_RUN_ROOTS = {
    "100% (set2)": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/qwen-2.5/output2/Qwen3VL_set2"
    ),
    "50%": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/qwen-2.5/output2/Qwen3-VL-8B-Instruct_p50"
    ),
    "60%": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/qwen-2.5/output2/Qwen3-VL-8B-Instruct_p60"
    ),
    "70%": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/qwen-2.5/output2/Qwen3-VL-8B-Instruct_p70"
    ),
    "80%": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/qwen-2.5/output2/Qwen3-VL-8B-Instruct_p80"
    ),
}

PIXTRAL_RUN_ROOTS = {
    "100% (set2)": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/pixtral/output2/pixtral-12b-set2"
    ),
    "50%": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/pixtral/output2/pixtral-12b_p50"
    ),
    "60%": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/pixtral/output2/pixtral-12b_p60"
    ),
    "70%": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/pixtral/output2/pixtral-12b_p70"
    ),
    "80%": Path(
        "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/pixtral/output2/pixtral-12b_p80"
    ),
}

QWEN_OUTPUT_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/manifold_analysis/multi_run_manifold_results"
)
QWEN_PAPER_PLOTS_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/plots_paper/qwen3vl_manifold"
)
PIXTRAL_OUTPUT_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/manifold_analysis/pixtral_multi_run_manifold_results"
)
PIXTRAL_PAPER_PLOTS_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/plots_paper/pixtral_manifold"
)

PRESETS = {
    "qwen": {
        "run_roots": QWEN_RUN_ROOTS,
        "output_dir": QWEN_OUTPUT_DIR,
        "paper_plots_dir": QWEN_PAPER_PLOTS_DIR,
    },
    "pixtral": {
        "run_roots": PIXTRAL_RUN_ROOTS,
        "output_dir": PIXTRAL_OUTPUT_DIR,
        "paper_plots_dir": PIXTRAL_PAPER_PLOTS_DIR,
    },
}

RUN_ROOTS = QWEN_RUN_ROOTS
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
    adapter_path: str
    step: Optional[int]
    epoch: Optional[float]
    loss: Optional[float]
    learning_rate: Optional[float]
    lora_scale: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def natural_sort_key(path: Path) -> Tuple[int, int, str]:
    m = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if m:
        return (0, int(m.group(1)), path.name)
    if path.name == "checkpoint-final":
        return (1, 10**18, path.name)
    return (2, 10**18, path.name)


def checkpoint_step(name: str) -> Optional[int]:
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else None


def trainer_state_log(state_path: Path, step: Optional[int]) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    with state_path.open() as f:
        state = json.load(f)
    history = [e for e in (state.get("log_history") or []) if isinstance(e, dict) and "loss" in e]
    if step is not None:
        exact = [e for e in history if e.get("step") == step]
        if exact:
            return exact[-1]
        before = [e for e in history if e.get("step", -1) <= step]
        if before:
            return before[-1]
    return history[-1] if history else {}


def lora_scale(config_path: Path) -> float:
    with config_path.open() as f:
        cfg = json.load(f)
    r = cfg.get("r", 1)
    alpha = cfg.get("lora_alpha", r)
    return float(alpha) / float(r) if r else 1.0


def discover_checkpoints(run_name: str, root: Path) -> List[CheckpointPoint]:
    dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")
         and (p / "adapter_model.safetensors").exists()],
        key=natural_sort_key,
    )
    points: List[CheckpointPoint] = []
    for d in dirs:
        step = checkpoint_step(d.name)
        scale = lora_scale(d / "adapter_config.json") if (d / "adapter_config.json").exists() else 2.0
        log = trainer_state_log(d / "trainer_state.json", step)
        points.append(CheckpointPoint(
            run_name=run_name,
            checkpoint_name=d.name,
            adapter_path=str(d / "adapter_model.safetensors"),
            step=step,
            epoch=log.get("epoch"),
            loss=log.get("loss"),
            learning_rate=log.get("learning_rate"),
            lora_scale=scale,
        ))
    return points


def extract_lora_modules(adapter_path: str) -> List[str]:
    with safe_open(adapter_path, framework="pt", device="cpu") as h:
        keys = set(h.keys())
    modules = sorted(
        k[: -len(LORA_A_SUFFIX)]
        for k in keys
        if k.endswith(LORA_A_SUFFIX) and (k[: -len(LORA_A_SUFFIX)] + LORA_B_SUFFIX) in keys
    )
    return modules


def low_rank_ip(A_i, B_i, A_j, B_j) -> float:
    import torch
    A_i = A_i.to(torch.float32)
    B_i = B_i.to(torch.float32)
    A_j = A_j.to(torch.float32)
    B_j = B_j.to(torch.float32)
    return float((B_i.T @ B_j * (A_j @ A_i.T)).sum().double().item())


def compute_kernel(points: List[CheckpointPoint], modules: List[str]) -> np.ndarray:
    n = len(points)
    kernel = np.zeros((n, n), dtype=np.float64)
    scales = np.array([p.lora_scale for p in points], dtype=np.float64)
    print(f"Computing exact kernel for {n} adapter checkpoints and {len(modules)} LoRA modules ...")
    with ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(p.adapter_path, framework="pt", device="cpu"))
            for p in points
        ]
        for mi, module in enumerate(modules, 1):
            a_key = module + LORA_A_SUFFIX
            b_key = module + LORA_B_SUFFIX
            As = [h.get_tensor(a_key) for h in handles]
            Bs = [h.get_tensor(b_key) for h in handles]
            for i in range(n):
                for j in range(i, n):
                    ip = low_rank_ip(As[i], Bs[i], As[j], Bs[j])
                    ip *= scales[i] * scales[j]
                    kernel[i, j] += ip
                    if j != i:
                        kernel[j, i] += ip
            if mi == 1 or mi == len(modules) or mi % 25 == 0:
                print(f"  {mi:>3}/{len(modules)} modules")
    return kernel


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    one = np.ones((n, n), dtype=np.float64) / n
    return K - one @ K - K @ one + one @ K @ one


def kpca_coords(K: np.ndarray, dims: int) -> Tuple[np.ndarray, np.ndarray]:
    Kc = center_kernel(K)
    evals, evecs = np.linalg.eigh(Kc)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    pos = np.maximum(evals[:dims], 0.0)
    coords = evecs[:, :dims] * np.sqrt(pos)[None, :]
    total = float(np.sum(np.maximum(evals, 0.0)))
    explained = pos / total if total > 0 else np.zeros(dims)
    return coords, explained


def norms_from_kernel(K: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.diag(K), 0.0))


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
    """Collapse adjacent identical checkpoints, e.g. checkpoint-final == final step."""
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


def padded_xy_limits(coords: np.ndarray, pad_frac: float = 0.16, min_pad: float = 0.75) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    xpad = max((xmax - xmin) * pad_frac, min_pad)
    ypad = max((ymax - ymin) * pad_frac, min_pad)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def add_paper_note(ax, text: str) -> None:
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.86),
        zorder=10,
    )


def annotate_2d_steps(ax, xy: np.ndarray, labels: Sequence[str], color: str = "black") -> None:
    offsets = [(0, -16), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10)]
    for k, (point, label) in enumerate(zip(xy, labels)):
        ann = ax.annotate(
            label,
            (point[0], point[1]),
            xytext=offsets[min(k, len(offsets) - 1)],
            textcoords="offset points",
            ha="center", va="center",
            fontsize=8.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.72),
            zorder=8,
        )
        ann.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def draw_highlight_trajectory_2d(ax, xy: np.ndarray, labels: Sequence[str], label: str = HIGHLIGHT_LABEL) -> None:
    ax.plot(xy[:, 0], xy[:, 1], "-", color="black", lw=5.2, alpha=0.45, zorder=5)
    ax.plot(
        xy[:, 0], xy[:, 1], "-o",
        color=HIGHLIGHT_COLOR,
        lw=3.4,
        ms=9.5,
        label=label,
        markerfacecolor=HIGHLIGHT_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.3,
        zorder=6,
    )
    for k in range(1, len(xy)):
        ax.annotate(
            "",
            xy=(xy[k, 0], xy[k, 1]),
            xytext=(xy[k - 1, 0], xy[k - 1, 1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=HIGHLIGHT_COLOR,
                lw=2.0,
                mutation_scale=16,
                shrinkA=15,
                shrinkB=15,
            ),
            zorder=7,
        )
    annotate_2d_steps(ax, xy, labels)


# ---------------------------------------------------------------------------
# RBF surface from checkpoint norms
# ---------------------------------------------------------------------------
def build_rbf_surface(
    coords_2d: np.ndarray,
    values: np.ndarray,
    grid_n: int = 180,
    pad_frac: float = 0.08,
    value_clip: Optional[Tuple[Optional[float], Optional[float]]] = None,
    mask_to_hull: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an RBF surface and evaluate on a regular grid."""
    xmin, xmax = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    ymin, ymax = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    xpad = (xmax - xmin) * pad_frac
    ypad = (ymax - ymin) * pad_frac
    xi = np.linspace(xmin - xpad, xmax + xpad, grid_n)
    yi = np.linspace(ymin - ypad, ymax + ypad, grid_n)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_pts = np.column_stack([Xi.ravel(), Yi.ravel()])

    rbf = RBFInterpolator(coords_2d, values, kernel="thin_plate_spline", smoothing=0.0)
    Zi = rbf(grid_pts).reshape(Xi.shape)
    if value_clip is not None:
        lo, hi = value_clip
        if lo is not None or hi is not None:
            Zi = np.clip(
                Zi,
                -np.inf if lo is None else lo,
                np.inf if hi is None else hi,
            )
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


def smooth_surface_trajectory(
    xy: np.ndarray,
    surface_rbf: RBFInterpolator,
    z_clip: Tuple[float, float],
    points_per_segment: int = 90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate the plotted path smoothly in x/y and evaluate z on the RBF surface."""
    if len(xy) < 2:
        z = np.asarray(surface_rbf(xy), dtype=np.float64)
        return xy.copy(), np.clip(z, z_clip[0], z_clip[1])

    chord = np.zeros(len(xy), dtype=np.float64)
    chord[1:] = np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    if len(xy) >= 4 and chord[-1] > 1e-12:
        t = chord / chord[-1]
        t_dense = np.linspace(0.0, 1.0, points_per_segment * (len(xy) - 1) + 1)
        sx = CubicSpline(t, xy[:, 0], bc_type="natural")
        sy = CubicSpline(t, xy[:, 1], bc_type="natural")
        smooth_xy = np.column_stack([sx(t_dense), sy(t_dense)])
    else:
        parts = []
        for k in range(len(xy) - 1):
            t_seg = np.linspace(0, 1, points_per_segment, endpoint=(k == len(xy) - 2))
            parts.append(np.column_stack([
                xy[k, 0] + t_seg * (xy[k + 1, 0] - xy[k, 0]),
                xy[k, 1] + t_seg * (xy[k + 1, 1] - xy[k, 1]),
            ]))
        smooth_xy = np.vstack(parts)

    smooth_z = np.asarray(surface_rbf(smooth_xy), dtype=np.float64)
    smooth_z = np.clip(smooth_z, z_clip[0], z_clip[1])
    return smooth_xy, smooth_z


# ---------------------------------------------------------------------------
# Main plot: genuine multi-run manifold surface
# ---------------------------------------------------------------------------
def plot_multi_run_manifold(
    all_points: List[CheckpointPoint],
    coords_2d: np.ndarray,
    norms: np.ndarray,
    base_coord: np.ndarray,
    explained: np.ndarray,
    output_dir: Path,
    paper_dir: Path,
    formats: Sequence[str],
) -> None:
    # Use all checkpoints to fit the smooth manifold surface, but plot only the
    # 100% trajectory requested for the paper figure.
    surface_coords = np.vstack([base_coord[None, :], coords_2d])
    surface_norms = np.concatenate([[0.0], norms])

    # Remove exact duplicates (same coordinates) before RBF
    seen = {}
    unique_idx = []
    for i, c in enumerate(surface_coords):
        key = (round(c[0], 10), round(c[1], 10))
        if key not in seen:
            seen[key] = i
            unique_idx.append(i)
    c2_unique = surface_coords[unique_idx]
    norms_unique = surface_norms[unique_idx]

    # Full extrapolated surface (no hull masking) for the 3D plot
    Xi_full, Yi_full, Zi_full = build_rbf_surface(
        c2_unique,
        norms_unique,
        grid_n=200,
        pad_frac=0.20,
        value_clip=(0.0, float(norms_unique.max()) * 1.10),
        mask_to_hull=False,
    )
    # Hull-masked version for the 2D contour (looks cleaner in 2D)
    Xi, Yi, Zi = build_rbf_surface(
        c2_unique,
        norms_unique,
        grid_n=220,
        value_clip=(0.0, float(norms_unique.max()) * 1.05),
        mask_to_hull=True,
    )
    Zi_masked = np.ma.masked_invalid(Zi)

    idx = highlighted_indices(all_points)
    traj_xy, traj_z, traj_labels = collapse_duplicate_trajectory_points(all_points, coords_2d, norms, idx)
    focus_xy = np.vstack([base_coord[None, :], traj_xy])
    xlim, ylim = padded_xy_limits(focus_xy, pad_frac=0.13, min_pad=0.85)

    # ---- 2D contour + trajectories ----
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    contour = ax.contourf(Xi, Yi, Zi_masked, levels=34, cmap="cividis", alpha=0.92)
    ax.contour(Xi, Yi, Zi_masked, levels=16, colors="black", linewidths=0.35, alpha=0.18)
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$||\Delta W||_F$ (LoRA update norm)")

    full_traj_xy_2d = np.vstack([base_coord[None, :], traj_xy])
    full_traj_labels_2d = ["base"] + list(traj_labels)
    draw_highlight_trajectory_2d(ax, full_traj_xy_2d, full_traj_labels_2d)

    # Base point
    ax.scatter(
        [base_coord[0]], [base_coord[1]],
        marker="*", s=360, color=BASE_COLOR, edgecolor="black", linewidth=1.2,
        zorder=9, label="base (pretrained)",
    )

    x_exp = explained[0] * 100
    y_exp = explained[1] * 100
    ax.set_xlabel(f"PC1 ({x_exp:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({y_exp:.1f}% variance)")
    ax.set_title("100% Fine-Tuning Trajectory on LoRA Weight Manifold\n(surface fitted from all 5 runs; only the 100% path is shown)")
    add_paper_note(ax, "Surface fit: 32 checkpoints from 5 runs\nPlotted trajectory: 100% fine-tuning run")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, "qwen3vl_multi_run_weight_manifold_2d", output_dir, paper_dir, formats)
    plt.close(fig)

    # ---- 3D surface + trajectory ----
    fig = plt.figure(figsize=(9.5, 7.0))
    ax3 = fig.add_subplot(111, projection="3d")

    # Surface height
    zmax_3d = max(float(traj_z.max()) * 1.20, 1.0)
    Zi_3d = np.clip(Zi_full, 0.0, zmax_3d)

    # Lit surface — keep it semi-transparent so the trajectory punches through
    light = LightSource(azdeg=315, altdeg=40)
    facecolors = light.shade(Zi_3d, cmap=cm.cividis, vert_exag=0.6, blend_mode="soft")
    facecolors[..., 3] = 0.50
    ax3.plot_surface(
        Xi_full, Yi_full, Zi_3d,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        rcount=160, ccount=160,
        shade=False,
        zorder=1,
    )

    # Dark contour lines on the surface for depth cues
    ax3.contour(Xi_full, Yi_full, Zi_3d, levels=16,
                colors="0.18", linewidths=0.50, alpha=0.32, zorder=2)
    # Floor shadow contour
    ax3.contourf(Xi_full, Yi_full, Zi_3d, levels=18, cmap="cividis",
                 alpha=0.18, zdir="z", offset=0.0, zorder=0)

    # ---- trajectory on surface ----
    rbf_traj = RBFInterpolator(c2_unique, norms_unique,
                               kernel="thin_plate_spline", smoothing=0.0)
    full_traj_xy = np.vstack([base_coord[None, :], traj_xy])
    full_traj_z = np.concatenate([[0.0], traj_z])
    full_traj_labels = ["base"] + list(traj_labels)

    smooth_xy, smooth_z = smooth_surface_trajectory(
        full_traj_xy, rbf_traj, (0.0, zmax_3d), points_per_segment=100)
    rx, ry, rz = smooth_xy[:, 0], smooth_xy[:, 1], smooth_z
    # Lift trajectory slightly above the surface so it is never hidden
    rz_lift = rz + zmax_3d * 0.012

    # Floor shadow
    ax3.plot(rx, ry, np.zeros_like(rz), "-", color="0.35", lw=1.8, alpha=0.22, zorder=3)
    # Thick dark outline then vivid red on top
    ax3.plot(rx, ry, rz_lift, "-", color="black", lw=7.0, alpha=0.60,
             solid_capstyle="round", zorder=5)
    ax3.plot(rx, ry, rz_lift, "-", color=HIGHLIGHT_COLOR, lw=4.2,
             solid_capstyle="round", label=HIGHLIGHT_LABEL, zorder=6)

    # Checkpoint dot markers (skip base)
    wp_z = np.clip(rbf_traj(full_traj_xy), 0.0, zmax_3d)
    wp_z_lift = wp_z + zmax_3d * 0.012
    ax3.scatter(full_traj_xy[1:, 0], full_traj_xy[1:, 1], wp_z_lift[1:],
                color=HIGHLIGHT_COLOR, s=78, edgecolor="white",
                linewidth=1.4, zorder=7, depthshade=False)

    # Step labels — stagger vertically to avoid overlap
    label_offsets = [0.05, 0.07, 0.05, 0.07, 0.05, 0.07, 0.09, 0.05, 0.07, 0.05]
    for k in range(len(full_traj_xy)):
        x, y, z = full_traj_xy[k, 0], full_traj_xy[k, 1], float(wp_z_lift[k])
        frac = label_offsets[k % len(label_offsets)]
        z_off = zmax_3d * frac
        txt = ax3.text(x, y, z + z_off, full_traj_labels[k], fontsize=8.2,
                       zorder=8, color="black", ha="center", va="bottom",
                       weight="bold")
        txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    # Base star
    ax3.scatter([base_coord[0]], [base_coord[1]], [0.0], marker="*", s=440,
                color=BASE_COLOR, edgecolor="black", linewidth=1.2, zorder=9,
                label="base (pretrained)", depthshade=False)

    # Axes
    ax3.set_xlabel(f"PC1 ({x_exp:.1f}% var.)", labelpad=10, fontsize=9.5)
    ax3.set_ylabel(f"PC2 ({y_exp:.1f}% var.)", labelpad=10, fontsize=9.5)
    ax3.set_zlabel(r"$\|\Delta W\|_F$", labelpad=8, fontsize=9.5)
    ax3.set_title("Fine-Tuning Trajectory on LoRA Weight Manifold", fontsize=12, pad=14)

    # Full surface extent so the manifold bowl is visible
    ax3.set_xlim(float(Xi_full.min()), float(Xi_full.max()))
    ax3.set_ylim(float(Yi_full.min()), float(Yi_full.max()))
    ax3.set_zlim(0.0, zmax_3d)

    # Camera
    ax3.view_init(elev=30, azim=-55)
    try:
        ax3.set_proj_type("persp", focal_length=0.82)
    except TypeError:
        ax3.set_proj_type("persp")
    ax3.set_box_aspect([1.3, 1.0, 0.52])

    ax3.legend(fontsize=8, loc="upper left", framealpha=0.90,
               borderpad=0.4, handlelength=2.0, edgecolor="0.70")
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.xaxis.pane.set_edgecolor("0.80")
    ax3.yaxis.pane.set_edgecolor("0.80")
    ax3.zaxis.pane.set_edgecolor("0.80")
    ax3.grid(True, alpha=0.16, linestyle="--")
    ax3.tick_params(labelsize=8)
    fig.subplots_adjust(left=-0.06, right=1.00, bottom=-0.03, top=0.93)
    save_fig(fig, "qwen3vl_multi_run_weight_manifold_3d", output_dir, paper_dir, formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss landscape surface (if loss data available)
# ---------------------------------------------------------------------------
def plot_loss_landscape(
    all_points: List[CheckpointPoint],
    coords_2d: np.ndarray,
    base_coord: np.ndarray,
    explained: np.ndarray,
    output_dir: Path,
    paper_dir: Path,
    formats: Sequence[str],
) -> None:
    """Plot the training loss as a smooth RBF surface if enough loss values exist."""
    has_loss = [(i, p) for i, p in enumerate(all_points) if p.loss is not None]
    if len(has_loss) < 4:
        print("  Skipping loss landscape: fewer than 4 checkpoints have recorded loss values.")
        return

    idx_loss = [i for i, _ in has_loss]
    c2_loss = coords_2d[idx_loss]
    loss_vals = np.array([p.loss for _, p in has_loss], dtype=np.float64)

    # Deduplicate coordinates
    seen = {}
    unique_idx = []
    for k, i in enumerate(idx_loss):
        key = (round(c2_loss[k - len(idx_loss) + len(idx_loss), 0] if False else c2_loss[k, 0], 10),
               round(c2_loss[k, 1], 10))
        if key not in seen:
            seen[key] = k
            unique_idx.append(k)
    c2_u = c2_loss[unique_idx]
    loss_u = loss_vals[unique_idx]

    if len(c2_u) < 4:
        print("  Skipping loss landscape: fewer than 4 unique coordinate positions have loss.")
        return

    loss_pad = max(float(loss_vals.max() - loss_vals.min()) * 0.04, 1e-3)
    Xi, Yi, Zi = build_rbf_surface(
        c2_u,
        loss_u,
        grid_n=220,
        value_clip=(max(0.0, float(loss_vals.min()) - loss_pad), float(loss_vals.max()) + loss_pad),
    )
    Zi_masked = np.ma.masked_invalid(Zi)

    hi_idx = highlighted_indices(all_points, require_loss=True)
    loss_by_point = np.array([np.nan if p.loss is None else p.loss for p in all_points], dtype=np.float64)
    traj_xy, traj_loss, traj_labels = collapse_duplicate_trajectory_points(all_points, coords_2d, loss_by_point, hi_idx)
    focus_xy = np.vstack([base_coord[None, :], traj_xy])
    xlim, ylim = padded_xy_limits(focus_xy, pad_frac=0.13, min_pad=0.85)
    x_exp = explained[0] * 100
    y_exp = explained[1] * 100

    # 3D loss surface
    Xi_full_loss, Yi_full_loss, Zi_full_loss = build_rbf_surface(
        c2_u, loss_u, grid_n=200, pad_frac=0.20,
        value_clip=(max(0.0, float(loss_vals.min()) - loss_pad),
                    float(loss_vals.max()) + loss_pad),
        mask_to_hull=False,
    )
    fig = plt.figure(figsize=(9.5, 7.2))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot_surface(Xi_full_loss, Yi_full_loss, Zi_full_loss, cmap="coolwarm_r",
                     alpha=0.42, linewidth=0, antialiased=True, zorder=1)

    # Shaded isolines on the surface for depth perspective
    ax3.contour(Xi_full_loss, Yi_full_loss, Zi_full_loss, levels=18,
                cmap="coolwarm_r", linewidths=0.7, alpha=0.6, zorder=2)
    # Filled contour projection on the floor for depth shading
    zfloor_loss = max(0.0, float(loss_vals.min()) - loss_pad)
    ax3.contourf(Xi_full_loss, Yi_full_loss, Zi_full_loss, levels=20,
                 cmap="coolwarm_r", alpha=0.30, zdir='z', offset=zfloor_loss, zorder=0)

    # Build RBF for interpolating trajectory along the surface
    rbf_loss = RBFInterpolator(c2_u, loss_u,
                               kernel="thin_plate_spline", smoothing=0.0)

    # Include base point in trajectory (loss estimated from surface)
    base_loss_est = float(np.clip(rbf_loss(base_coord[None, :]), 0.0, None)[0])
    full_traj_xy = np.vstack([base_coord[None, :], traj_xy])
    full_traj_loss = np.concatenate([[base_loss_est], traj_loss])
    full_traj_labels = ["base"] + list(traj_labels)

    # Densely interpolate trajectory along the surface
    n_seg = 60
    smooth_parts = []
    for k in range(len(full_traj_xy) - 1):
        t = np.linspace(0, 1, n_seg, endpoint=(k == len(full_traj_xy) - 2))
        seg = np.column_stack([
            full_traj_xy[k, 0] + t * (full_traj_xy[k+1, 0] - full_traj_xy[k, 0]),
            full_traj_xy[k, 1] + t * (full_traj_xy[k+1, 1] - full_traj_xy[k, 1]),
        ])
        smooth_parts.append(seg)
    smooth_xy = np.vstack(smooth_parts)
    smooth_z = rbf_loss(smooth_xy)

    rx, ry, rz = smooth_xy[:, 0], smooth_xy[:, 1], smooth_z
    # Shadow on floor
    ax3.plot(rx, ry, np.full_like(rz, zfloor_loss), "-", color="gray",
             lw=1.6, alpha=0.30, zorder=3)
    # Main trajectory curving along the surface
    ax3.plot(rx, ry, rz, "-", color="black", lw=5.5, alpha=0.50,
             solid_capstyle="round", zorder=5)
    ax3.plot(rx, ry, rz, "-", color=HIGHLIGHT_COLOR, lw=3.4,
             label=HIGHLIGHT_LABEL, zorder=6)
    # Waypoint markers at original checkpoint positions (on surface)
    wp_z = rbf_loss(full_traj_xy)
    ax3.scatter(full_traj_xy[:, 0], full_traj_xy[:, 1], wp_z,
                color=HIGHLIGHT_COLOR, s=80, edgecolor="white",
                linewidth=1.4, zorder=7, depthshade=False)
    # Base point star marker
    ax3.scatter([base_coord[0]], [base_coord[1]], [base_loss_est], marker="*",
                s=400, color=BASE_COLOR, edgecolor="black", linewidth=1.2,
                zorder=9, label="base (pretrained)", depthshade=False)
    # Step labels
    for k in range(len(full_traj_xy)):
        x, y, z = full_traj_xy[k, 0], full_traj_xy[k, 1], float(wp_z[k])
        z_off = max(float(np.abs(full_traj_loss.max() - full_traj_loss.min())) * 0.04, 0.01)
        txt = ax3.text(x, y, z + z_off, f" {full_traj_labels[k]}", fontsize=8.5,
                       zorder=8, color="black", ha="center", va="bottom",
                       weight="semibold")
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    ax3.set_xlabel(f"PC1 ({x_exp:.1f}%)", labelpad=10)
    ax3.set_ylabel(f"PC2 ({y_exp:.1f}%)", labelpad=10)
    ax3.set_zlabel("Training Loss", labelpad=10)
    ax3.set_title("100% Training Trajectory on LoRA Loss Surface\n(loss surface fitted from all checkpoint losses)")
    ax3.set_xlim(float(Xi_full_loss.min()), float(Xi_full_loss.max()))
    ax3.set_ylim(float(Yi_full_loss.min()), float(Yi_full_loss.max()))
    zpad = max(float(full_traj_loss.max() - full_traj_loss.min()) * 0.18, 0.03)
    ax3.set_zlim(max(0.0, float(full_traj_loss.min()) - zpad), float(full_traj_loss.max()) + zpad)
    ax3.view_init(elev=26, azim=-63)
    ax3.set_box_aspect([1.55, 0.75, 0.60])
    ax3.legend(fontsize=8.5, loc="upper left")
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.grid(True, alpha=0.22, linestyle="--")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94)
    save_fig(fig, "qwen3vl_multi_run_loss_landscape_3d", output_dir, paper_dir, formats)
    plt.close(fig)

    # 2D contour loss
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
    ax.set_title("100% Fine-Tuning Trajectory on LoRA Loss Surface\n(surface fitted from all 5 runs; only the 100% path is shown)")
    add_paper_note(ax, "Loss surface fit: all checkpoints with loss\nPlotted trajectory: 100% fine-tuning run")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, "qwen3vl_multi_run_loss_landscape_2d", output_dir, paper_dir, formats)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-run LoRA manifold surface")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--paper-plots-dir", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--skip-plots", action="store_true", help="Only compute and save manifold data files")
    return parser.parse_args()


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    run_roots = preset["run_roots"]
    output_dir = (args.output_dir or preset["output_dir"]).resolve()
    paper_dir = (args.paper_plots_dir or preset["paper_plots_dir"]).resolve()

    # 1. Discover all checkpoints
    all_points: List[CheckpointPoint] = []
    for run_name, root in sorted(run_roots.items(), key=lambda kv: kv[0]):
        if not root.exists():
            print(f"  Skipping {run_name}: {root} not found")
            continue
        pts = discover_checkpoints(run_name, root)
        print(f"  {run_name}: {len(pts)} checkpoints")
        all_points.extend(pts)
    print(f"Total adapter checkpoints: {len(all_points)}")

    if len(all_points) < 3:
        print("ERROR: need at least 3 checkpoints across all runs.")
        return

    # 2. Validate modules match
    modules = extract_lora_modules(all_points[0].adapter_path)
    print(f"LoRA modules: {len(modules)}")

    # 3. Compute kernel (adapters only, no base row yet)
    kernel_adapters = compute_kernel(all_points, modules)

    # 4. Prepend base as zero-update point: kernel row/col of zeros
    n = len(all_points)
    kernel = np.zeros((n + 1, n + 1), dtype=np.float64)
    kernel[1:, 1:] = kernel_adapters

    # Norms include base (0)
    norms_full = np.sqrt(np.maximum(np.diag(kernel), 0.0))

    # 5. Kernel PCA (2D)
    coords, explained = kpca_coords(kernel, dims=2)
    # Separate base from adapters
    base_coord = coords[0]
    adapter_coords = coords[1:]
    adapter_norms = norms_full[1:]

    print(f"PC1 explains {explained[0]*100:.1f}%, PC2 explains {explained[1]*100:.1f}%")

    # 6. Save data — enough for the standalone plotting script to work
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "multi_run_kernel.npy", kernel)
    np.save(output_dir / "multi_run_coords_2d.npy", coords)
    np.save(output_dir / "multi_run_explained.npy", explained)
    np.save(output_dir / "multi_run_base_coord.npy", base_coord)
    np.save(output_dir / "multi_run_adapter_coords.npy", adapter_coords)
    np.save(output_dir / "multi_run_adapter_norms.npy", adapter_norms)
    manifest = []
    manifest.append({"index": 0, "run": "base", "checkpoint": "base", "step": 0, "loss": None,
                     "norm": 0.0, "pc1": float(base_coord[0]), "pc2": float(base_coord[1])})
    for i, p in enumerate(all_points):
        manifest.append({
            "index": i + 1, "run": p.run_name, "checkpoint": p.checkpoint_name,
            "step": p.step, "epoch": p.epoch, "loss": p.loss, "lr": p.learning_rate,
            "norm": float(adapter_norms[i]),
            "pc1": float(adapter_coords[i, 0]), "pc2": float(adapter_coords[i, 1]),
        })
    # Also save the CheckpointPoint list as JSON for the plotting script
    points_data = []
    for p in all_points:
        points_data.append({
            "run_name": p.run_name, "checkpoint_name": p.checkpoint_name,
            "step": p.step, "epoch": p.epoch, "loss": p.loss,
            "learning_rate": p.learning_rate, "lora_scale": p.lora_scale,
        })
    with (output_dir / "multi_run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with (output_dir / "multi_run_points.json").open("w") as f:
        json.dump(points_data, f, indent=2, default=str)
    print(f"Data saved to {output_dir}")

    # 7. Plots
    if not args.skip_plots:
        plot_multi_run_manifold(all_points, adapter_coords, adapter_norms, base_coord, explained,
                                output_dir, paper_dir, args.formats)
        plot_loss_landscape(all_points, adapter_coords, base_coord, explained,
                            output_dir, paper_dir, args.formats)

    print(f"\nDone.  Data: {output_dir}")
    print(f"       Plots: {paper_dir}")


if __name__ == "__main__":
    main()
