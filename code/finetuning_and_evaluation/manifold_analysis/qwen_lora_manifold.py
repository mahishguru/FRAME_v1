#!/usr/bin/env python3
"""Map a Qwen3-VL LoRA fine-tuning run as a trajectory in weight-update space.

The script treats every saved PEFT/LoRA checkpoint as a point whose coordinates are
not the raw LoRA factors, but the effective update induced by each adapter:

    Delta W_l(t) = (alpha / r) * B_l(t) @ A_l(t)

Pairwise checkpoint geometry is computed exactly with a low-rank Frobenius kernel,
without loading the base 8B model and without materializing dense Delta W matrices.
The resulting kernel/distances are projected to 2D/3D for paper-ready trajectory
figures and saved with quantitative path diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import safe_open

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CHECKPOINT_ROOT = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/qwen-2.5/output2/Qwen3VL_set2"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/manifold_analysis/qwen3vl_set2_results"
)
DEFAULT_PAPER_PLOTS_DIR = Path(
    "/data/mguru/04_Finetuning/fine_tune_llm_post_processing/evaluation/plots_paper/qwen3vl_manifold"
)

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


@dataclass
class CheckpointEntry:
    """Metadata for one point on the trajectory."""

    index: int
    name: str
    label: str
    kind: str
    path: Optional[str]
    adapter_path: Optional[str]
    step: Optional[int]
    epoch: Optional[float]
    loss: Optional[float]
    learning_rate: Optional[float]
    grad_norm: Optional[float]
    has_trainer_state: bool
    lora_r: Optional[int]
    lora_alpha: Optional[float]
    lora_scale: Optional[float]
    target_modules: List[str]


@dataclass
class ValidationReport:
    kernel_symmetric_max_abs: float
    kernel_min_eigenvalue: float
    kernel_psd_tolerance_ok: bool
    min_distance: float
    max_distance: float
    diagonal_distance_max_abs: float
    low_rank_materialization_abs_error: Optional[float]
    low_rank_materialization_rel_error: Optional[float]
    duplicate_previous_distance: Optional[float]
    final_duplicates_previous: Optional[bool]


@dataclass
class PathSummary:
    n_points: int
    n_non_base_checkpoints: int
    endpoint_label: str
    endpoint_displacement: float
    cumulative_path_length_all: float
    cumulative_path_length_nonduplicate: float
    straightness_all: float
    straightness_nonduplicate: float
    duplicate_tolerance: float
    consecutive_duplicate_pairs: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot a Qwen3-VL LoRA checkpoint manifold trajectory."
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"Directory containing checkpoint-* PEFT adapter folders (default: {DEFAULT_CHECKPOINT_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for data outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--paper-plots-dir",
        type=Path,
        default=DEFAULT_PAPER_PLOTS_DIR,
        help=f"Directory for paper-ready figures (default: {DEFAULT_PAPER_PLOTS_DIR})",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Do not prepend the pretrained model as a zero-update base point.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to save.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of embedding dimensions to save for kernel PCA/MDS coordinates.",
    )
    parser.add_argument(
        "--duplicate-tolerance",
        type=float,
        default=1e-8,
        help="Distance threshold for marking consecutive checkpoints as duplicates.",
    )
    parser.add_argument(
        "--skip-validation-materialization",
        action="store_true",
        help="Skip explicit BA materialization validation on the smallest module.",
    )
    return parser.parse_args()


def natural_checkpoint_sort_key(path: Path) -> Tuple[int, int, str]:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    if path.name == "checkpoint-final":
        return (1, 10**18, path.name)
    return (2, 10**18, path.name)


def discover_checkpoint_dirs(checkpoint_root: Path) -> List[Path]:
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")
    checkpoint_dirs = [p for p in checkpoint_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    checkpoint_dirs = [p for p in checkpoint_dirs if (p / "adapter_model.safetensors").exists()]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint adapter_model.safetensors files found in {checkpoint_root}")
    return sorted(checkpoint_dirs, key=natural_checkpoint_sort_key)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_training_log(trainer_state: Dict[str, Any], fallback_step: Optional[int]) -> Dict[str, Any]:
    history = trainer_state.get("log_history", []) or []
    candidates = [entry for entry in history if isinstance(entry, dict) and "loss" in entry]
    if fallback_step is not None:
        exact = [entry for entry in candidates if entry.get("step") == fallback_step]
        if exact:
            return exact[-1]
        before = [entry for entry in candidates if entry.get("step", -1) <= fallback_step]
        if before:
            return before[-1]
    return candidates[-1] if candidates else {}


def numeric_step_from_name(name: str) -> Optional[int]:
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    return int(match.group(1)) if match else None


def lora_scale_from_config(config: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[float], List[str]]:
    r = config.get("r")
    alpha = config.get("lora_alpha")
    scale = None
    if r not in (None, 0) and alpha is not None:
        scale = float(alpha) / float(r)
    target_modules = sorted(config.get("target_modules") or [])
    return r, alpha, scale, target_modules


def build_manifest(checkpoint_dirs: Sequence[Path], include_base: bool) -> List[CheckpointEntry]:
    entries: List[CheckpointEntry] = []
    if include_base:
        entries.append(
            CheckpointEntry(
                index=0,
                name="base",
                label="base",
                kind="base_zero_update",
                path=None,
                adapter_path=None,
                step=0,
                epoch=0.0,
                loss=None,
                learning_rate=None,
                grad_norm=None,
                has_trainer_state=False,
                lora_r=None,
                lora_alpha=None,
                lora_scale=None,
                target_modules=[],
            )
        )

    for checkpoint_dir in checkpoint_dirs:
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        config = load_json(adapter_config_path) if adapter_config_path.exists() else {}
        r, alpha, scale, target_modules = lora_scale_from_config(config)

        step = numeric_step_from_name(checkpoint_dir.name)
        epoch = None
        loss = None
        lr = None
        grad_norm = None
        has_trainer_state = trainer_state_path.exists()
        if has_trainer_state:
            trainer_state = load_json(trainer_state_path)
            step = int(trainer_state.get("global_step") or step or 0)
            epoch = _to_optional_float(trainer_state.get("epoch"))
            latest_log = latest_training_log(trainer_state, step)
            loss = _to_optional_float(latest_log.get("loss"))
            lr = _to_optional_float(latest_log.get("learning_rate"))
            grad_norm = _to_optional_float(latest_log.get("grad_norm"))

        label = checkpoint_dir.name.replace("checkpoint-", "")
        entries.append(
            CheckpointEntry(
                index=len(entries),
                name=checkpoint_dir.name,
                label=label,
                kind="lora_adapter_checkpoint",
                path=str(checkpoint_dir),
                adapter_path=str(checkpoint_dir / "adapter_model.safetensors"),
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=lr,
                grad_norm=grad_norm,
                has_trainer_state=has_trainer_state,
                lora_r=r,
                lora_alpha=alpha,
                lora_scale=scale,
                target_modules=target_modules,
            )
        )
    return entries


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def adapter_entries(manifest: Sequence[CheckpointEntry]) -> List[CheckpointEntry]:
    return [entry for entry in manifest if entry.adapter_path is not None]


def read_safetensor_keys(adapter_path: Path) -> List[str]:
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        return list(handle.keys())


def extract_lora_modules(keys: Iterable[str]) -> List[str]:
    modules = []
    key_set = set(keys)
    for key in sorted(key_set):
        if key.endswith(LORA_A_SUFFIX):
            module = key[: -len(LORA_A_SUFFIX)]
            b_key = module + LORA_B_SUFFIX
            if b_key not in key_set:
                raise KeyError(f"Found {key}, but missing paired {b_key}")
            modules.append(module)
    if not modules:
        raise ValueError("No LoRA A/B weight pairs found.")
    return modules


def tensor_shape(handle: Any, key: str) -> Tuple[int, ...]:
    tensor = handle.get_tensor(key)
    return tuple(int(x) for x in tensor.shape)


def validate_adapter_key_shapes(entries: Sequence[CheckpointEntry]) -> List[str]:
    adapters = adapter_entries(entries)
    if not adapters:
        raise ValueError("No adapter checkpoints found after manifest construction.")
    reference_path = Path(adapters[0].adapter_path or "")
    reference_keys = read_safetensor_keys(reference_path)
    modules = extract_lora_modules(reference_keys)
    reference_key_set = set(reference_keys)

    with safe_open(str(reference_path), framework="pt", device="cpu") as ref_handle:
        reference_shapes = {key: tensor_shape(ref_handle, key) for key in reference_keys}

    for entry in adapters[1:]:
        adapter_path = Path(entry.adapter_path or "")
        with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            key_set = set(keys)
            if key_set != reference_key_set:
                missing = sorted(reference_key_set - key_set)[:10]
                extra = sorted(key_set - reference_key_set)[:10]
                raise ValueError(
                    f"Adapter keys do not match for {entry.name}. Missing={missing}; extra={extra}"
                )
            for key in reference_keys:
                shape = tensor_shape(handle, key)
                if shape != reference_shapes[key]:
                    raise ValueError(
                        f"Shape mismatch for {entry.name}:{key}: expected {reference_shapes[key]}, got {shape}"
                    )
    return modules


def module_type(module_name: str) -> str:
    return module_name.split(".")[-1]


def layer_id(module_name: str) -> str:
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    if match:
        return f"layer_{int(match.group(1)):02d}"
    return "unknown_layer"


def block_type(module_name: str) -> str:
    if ".self_attn." in module_name:
        return "self_attn"
    if ".mlp." in module_name:
        return "mlp"
    if ".visual." in module_name or ".vision" in module_name:
        return "vision"
    return "other"


def module_short_name(module_name: str) -> str:
    layer = layer_id(module_name)
    block = block_type(module_name)
    mod = module_type(module_name)
    return f"{layer}.{block}.{mod}"


def entry_scale(entry: CheckpointEntry) -> float:
    if entry.lora_scale is None:
        raise ValueError(f"Checkpoint {entry.name} has no LoRA scale in adapter_config.json")
    return float(entry.lora_scale)


def low_rank_inner_product(A_i: torch.Tensor, B_i: torch.Tensor, A_j: torch.Tensor, B_j: torch.Tensor) -> float:
    """Return <B_i A_i, B_j A_j>_F without materializing dense products."""
    A_i_f = A_i.to(dtype=torch.float32)
    B_i_f = B_i.to(dtype=torch.float32)
    A_j_f = A_j.to(dtype=torch.float32)
    B_j_f = B_j.to(dtype=torch.float32)
    b_gram = B_i_f.transpose(0, 1).matmul(B_j_f)
    a_gram = A_i_f.matmul(A_j_f.transpose(0, 1))
    return float(torch.sum(b_gram * a_gram).double().item())


def compute_low_rank_kernel(
    manifest: Sequence[CheckpointEntry], modules: Sequence[str]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    adapters = adapter_entries(manifest)
    n_adapters = len(adapters)
    n_points = len(manifest)
    base_offset = n_points - n_adapters
    if base_offset not in (0, 1):
        raise ValueError("Only zero or one prepended base point is supported.")

    total_kernel = np.zeros((n_points, n_points), dtype=np.float64)
    module_kernels: Dict[str, np.ndarray] = {}
    adapter_paths = [Path(entry.adapter_path or "") for entry in adapters]
    scales = np.asarray([entry_scale(entry) for entry in adapters], dtype=np.float64)

    print(f"Computing exact low-rank kernel for {n_adapters} checkpoints and {len(modules)} LoRA modules...")
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(path), framework="pt", device="cpu")) for path in adapter_paths]
        for module_idx, module in enumerate(modules, start=1):
            a_key = module + LORA_A_SUFFIX
            b_key = module + LORA_B_SUFFIX
            a_tensors = [handle.get_tensor(a_key) for handle in handles]
            b_tensors = [handle.get_tensor(b_key) for handle in handles]
            local_kernel = np.zeros((n_adapters, n_adapters), dtype=np.float64)
            for i in range(n_adapters):
                for j in range(i, n_adapters):
                    ip = low_rank_inner_product(a_tensors[i], b_tensors[i], a_tensors[j], b_tensors[j])
                    ip *= scales[i] * scales[j]
                    local_kernel[i, j] = ip
                    local_kernel[j, i] = ip
            expanded = np.zeros((n_points, n_points), dtype=np.float64)
            expanded[base_offset:, base_offset:] = local_kernel
            module_kernels[module] = expanded
            total_kernel += expanded
            if module_idx == 1 or module_idx == len(modules) or module_idx % 25 == 0:
                print(f"  processed {module_idx:>3}/{len(modules)} modules")
    return total_kernel, module_kernels


def squared_distances_from_kernel(kernel: np.ndarray) -> np.ndarray:
    diag = np.diag(kernel)
    d2 = diag[:, None] + diag[None, :] - 2.0 * kernel
    d2[np.abs(d2) < 1e-10] = 0.0
    return np.maximum(d2, 0.0)


def angular_distances_from_kernel(kernel: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.maximum(np.diag(kernel), 0.0))
    n = kernel.shape[0]
    angular = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if norms[i] > 0 and norms[j] > 0:
                cos = kernel[i, j] / (norms[i] * norms[j])
                angular[i, j] = math.acos(float(np.clip(cos, -1.0, 1.0)))
            elif i == j:
                angular[i, j] = 0.0
    return angular


def center_kernel(kernel: np.ndarray) -> np.ndarray:
    n = kernel.shape[0]
    one = np.ones((n, n), dtype=np.float64) / n
    return kernel - one @ kernel - kernel @ one + one @ kernel @ one


def coordinates_from_kernel(kernel: np.ndarray, dims: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = center_kernel(kernel)
    evals, evecs = np.linalg.eigh(centered)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    positive = np.maximum(evals, 0.0)
    coords = evecs[:, :dims] * np.sqrt(positive[:dims])[None, :]
    if coords.shape[1] < dims:
        coords = np.pad(coords, ((0, 0), (0, dims - coords.shape[1])), constant_values=0.0)
    total_positive = float(np.sum(positive))
    explained = positive[:dims] / total_positive if total_positive > 0 else np.zeros(dims)
    return coords, evals, explained


def classical_mds_from_distances(distances: np.ndarray, dims: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d2 = np.square(distances)
    n = d2.shape[0]
    one = np.ones((n, n), dtype=np.float64) / n
    centered = -0.5 * ((np.eye(n) - one) @ d2 @ (np.eye(n) - one))
    return coordinates_from_kernel(centered, dims)


def spherical_mds_coordinates(
    angular_distances: np.ndarray, manifest: Sequence[CheckpointEntry], dims: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    nonzero_indices = [i for i, entry in enumerate(manifest) if entry.adapter_path is not None]
    angular_sub = angular_distances[np.ix_(nonzero_indices, nonzero_indices)]
    if not np.all(np.isfinite(angular_sub)):
        raise ValueError("Angular distance matrix for non-base checkpoints contains non-finite values.")
    coords_sub, evals, explained = classical_mds_from_distances(angular_sub, dims=dims)
    coords = np.full((len(manifest), dims), np.nan, dtype=np.float64)
    coords[nonzero_indices, :] = coords_sub
    return coords, evals, explained, nonzero_indices


def matrix_to_csv(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[_csv_float(x) for x in row]])


def records_to_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})


def _csv_float(value: Any) -> Any:
    if value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(value_f):
        return ""
    return f"{value_f:.12g}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return _csv_float(value) if isinstance(value, (float, np.floating)) else value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2)


def coordinate_records(
    manifest: Sequence[CheckpointEntry], coords: np.ndarray, prefix: str, explained: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    records = []
    for i, entry in enumerate(manifest):
        record = {
            "index": entry.index,
            "name": entry.name,
            "label": entry.label,
            "step": entry.step,
            "epoch": entry.epoch,
            "loss": entry.loss,
            "learning_rate": entry.learning_rate,
            "grad_norm": entry.grad_norm,
        }
        for dim_idx in range(coords.shape[1]):
            record[f"{prefix}{dim_idx + 1}"] = float(coords[i, dim_idx]) if np.isfinite(coords[i, dim_idx]) else None
        if explained is not None:
            for dim_idx in range(min(coords.shape[1], len(explained))):
                record[f"explained_var_{dim_idx + 1}"] = float(explained[dim_idx])
        records.append(record)
    return records


def inner_product_between_steps(kernel: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """Return <x_b - x_a, x_d - x_c> using only the Gram kernel."""
    return float(kernel[b, d] - kernel[b, c] - kernel[a, d] + kernel[a, c])


def compute_trajectory_metrics(
    manifest: Sequence[CheckpointEntry],
    kernel: np.ndarray,
    distances: np.ndarray,
    duplicate_tolerance: float,
) -> Tuple[PathSummary, List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels = [entry.label for entry in manifest]
    norms = np.sqrt(np.maximum(np.diag(kernel), 0.0))
    final_idx = len(manifest) - 1
    final_norm = norms[final_idx]
    final_label = labels[final_idx]

    point_records: List[Dict[str, Any]] = []
    cumulative = 0.0
    cumulative_nonduplicate = 0.0
    duplicate_pairs: List[str] = []
    step_distances = [0.0]
    for i in range(1, len(manifest)):
        step_distance = float(distances[i - 1, i])
        step_distances.append(step_distance)
        cumulative += step_distance
        if step_distance <= duplicate_tolerance:
            duplicate_pairs.append(f"{labels[i - 1]}->{labels[i]}")
        else:
            cumulative_nonduplicate += step_distance

    for i, entry in enumerate(manifest):
        align = None
        if norms[i] > 0 and final_norm > 0:
            align = float(np.clip(kernel[i, final_idx] / (norms[i] * final_norm), -1.0, 1.0))
        point_records.append(
            {
                "index": i,
                "name": entry.name,
                "label": entry.label,
                "step": entry.step,
                "epoch": entry.epoch,
                "loss": entry.loss,
                "learning_rate": entry.learning_rate,
                "grad_norm": entry.grad_norm,
                "radial_norm": float(norms[i]),
                "distance_from_base": float(distances[0, i]) if len(manifest) else 0.0,
                "distance_from_previous": float(step_distances[i]),
                "cumulative_path_length": float(sum(step_distances[: i + 1])),
                "cosine_alignment_with_final": align,
            }
        )

    segment_records: List[Dict[str, Any]] = []
    for i in range(1, len(manifest)):
        turn_cos = None
        turn_angle = None
        if 1 < i < len(manifest):
            prev_norm = distances[i - 2, i - 1]
            curr_norm = distances[i - 1, i]
            if prev_norm > 0 and curr_norm > 0:
                ip = inner_product_between_steps(kernel, i - 2, i - 1, i - 1, i)
                turn_cos = float(np.clip(ip / (prev_norm * curr_norm), -1.0, 1.0))
                turn_angle = float(math.degrees(math.acos(turn_cos)))
        segment_records.append(
            {
                "segment_index": i,
                "from_label": labels[i - 1],
                "to_label": labels[i],
                "from_step": manifest[i - 1].step,
                "to_step": manifest[i].step,
                "distance": float(distances[i - 1, i]),
                "is_duplicate": bool(distances[i - 1, i] <= duplicate_tolerance),
                "turn_cosine_with_previous_segment": turn_cos,
                "turn_angle_degrees_with_previous_segment": turn_angle,
            }
        )

    endpoint_displacement = float(distances[0, final_idx]) if len(manifest) > 1 else 0.0
    summary = PathSummary(
        n_points=len(manifest),
        n_non_base_checkpoints=len(adapter_entries(manifest)),
        endpoint_label=final_label,
        endpoint_displacement=endpoint_displacement,
        cumulative_path_length_all=float(cumulative),
        cumulative_path_length_nonduplicate=float(cumulative_nonduplicate),
        straightness_all=float(endpoint_displacement / cumulative) if cumulative > 0 else 0.0,
        straightness_nonduplicate=float(endpoint_displacement / cumulative_nonduplicate)
        if cumulative_nonduplicate > 0
        else 0.0,
        duplicate_tolerance=float(duplicate_tolerance),
        consecutive_duplicate_pairs=duplicate_pairs,
    )
    return summary, point_records, segment_records


def grouped_norm_records(
    manifest: Sequence[CheckpointEntry], module_kernels: Dict[str, np.ndarray], group: str
) -> List[Dict[str, Any]]:
    final_idx = len(manifest) - 1
    groups: Dict[str, float] = {}
    for module, module_kernel in module_kernels.items():
        if group == "module_type":
            key = module_type(module)
        elif group == "layer":
            key = layer_id(module)
        elif group == "block_type":
            key = block_type(module)
        else:
            key = module_short_name(module)
        groups[key] = groups.get(key, 0.0) + float(max(module_kernel[final_idx, final_idx], 0.0))
    total = sum(groups.values())
    records = []
    for key, norm_sq in sorted(groups.items(), key=lambda item: item[1], reverse=True):
        records.append(
            {
                group: key,
                "final_norm_squared": norm_sq,
                "final_norm": math.sqrt(max(norm_sq, 0.0)),
                "fraction_of_final_update_norm_squared": norm_sq / total if total > 0 else 0.0,
            }
        )
    return records


def grouped_step_movement_records(
    manifest: Sequence[CheckpointEntry], module_kernels: Dict[str, np.ndarray], group: str
) -> List[Dict[str, Any]]:
    labels = [entry.label for entry in manifest]
    records = []
    for i in range(1, len(manifest)):
        groups: Dict[str, float] = {}
        for module, module_kernel in module_kernels.items():
            if group == "module_type":
                key = module_type(module)
            elif group == "layer":
                key = layer_id(module)
            elif group == "block_type":
                key = block_type(module)
            else:
                key = module_short_name(module)
            d2 = module_kernel[i - 1, i - 1] + module_kernel[i, i] - 2.0 * module_kernel[i - 1, i]
            groups[key] = groups.get(key, 0.0) + float(max(d2, 0.0))
        total = sum(groups.values())
        for key, d2 in sorted(groups.items()):
            records.append(
                {
                    "segment": f"{labels[i - 1]}->{labels[i]}",
                    "segment_index": i,
                    "from_label": labels[i - 1],
                    "to_label": labels[i],
                    group: key,
                    "movement_norm_squared": d2,
                    "movement_norm": math.sqrt(max(d2, 0.0)),
                    "fraction_of_segment_movement_norm_squared": d2 / total if total > 0 else 0.0,
                }
            )
    return records


def validate_low_rank_materialization(
    manifest: Sequence[CheckpointEntry], modules: Sequence[str], module_kernels: Dict[str, np.ndarray]
) -> Tuple[Optional[float], Optional[float]]:
    adapters = adapter_entries(manifest)
    if len(adapters) < 2:
        return None, None
    # Pick the module with the smallest dense BA footprint for validation.
    best_module = None
    best_size = None
    first_path = Path(adapters[0].adapter_path or "")
    with safe_open(str(first_path), framework="pt", device="cpu") as handle:
        for module in modules:
            a_shape = tensor_shape(handle, module + LORA_A_SUFFIX)
            b_shape = tensor_shape(handle, module + LORA_B_SUFFIX)
            dense_size = b_shape[0] * a_shape[1]
            if best_size is None or dense_size < best_size:
                best_size = dense_size
                best_module = module
    if best_module is None:
        return None, None

    i = 0
    j = 1
    entry_i = adapters[i]
    entry_j = adapters[j]
    with safe_open(str(entry_i.adapter_path), framework="pt", device="cpu") as hi, safe_open(
        str(entry_j.adapter_path), framework="pt", device="cpu"
    ) as hj:
        A_i = hi.get_tensor(best_module + LORA_A_SUFFIX).to(torch.float32)
        B_i = hi.get_tensor(best_module + LORA_B_SUFFIX).to(torch.float32)
        A_j = hj.get_tensor(best_module + LORA_A_SUFFIX).to(torch.float32)
        B_j = hj.get_tensor(best_module + LORA_B_SUFFIX).to(torch.float32)
        dense_i = B_i.matmul(A_i) * entry_scale(entry_i)
        dense_j = B_j.matmul(A_j) * entry_scale(entry_j)
        explicit = float(torch.sum(dense_i * dense_j).double().item())
    offset = len(manifest) - len(adapters)
    low_rank = float(module_kernels[best_module][offset + i, offset + j])
    abs_error = abs(explicit - low_rank)
    rel_error = abs_error / max(abs(explicit), 1e-12)
    return abs_error, rel_error


def build_validation_report(
    kernel: np.ndarray,
    distances: np.ndarray,
    manifest: Sequence[CheckpointEntry],
    modules: Sequence[str],
    module_kernels: Dict[str, np.ndarray],
    duplicate_tolerance: float,
    skip_materialization: bool,
) -> ValidationReport:
    sym = float(np.max(np.abs(kernel - kernel.T)))
    evals = np.linalg.eigvalsh((kernel + kernel.T) / 2.0)
    min_eval = float(np.min(evals))
    diag_max = float(np.max(np.abs(np.diag(distances))))
    material_abs = None
    material_rel = None
    if not skip_materialization:
        material_abs, material_rel = validate_low_rank_materialization(manifest, modules, module_kernels)
    duplicate_distance = None
    duplicates_previous = None
    if len(manifest) >= 2 and manifest[-1].name == "checkpoint-final":
        duplicate_distance = float(distances[-2, -1])
        duplicates_previous = bool(duplicate_distance <= duplicate_tolerance)
    return ValidationReport(
        kernel_symmetric_max_abs=sym,
        kernel_min_eigenvalue=min_eval,
        kernel_psd_tolerance_ok=bool(min_eval >= -1e-5 * max(1.0, float(np.max(np.diag(kernel))))),
        min_distance=float(np.min(distances)),
        max_distance=float(np.max(distances)),
        diagonal_distance_max_abs=diag_max,
        low_rank_materialization_abs_error=material_abs,
        low_rank_materialization_rel_error=material_rel,
        duplicate_previous_distance=duplicate_distance,
        final_duplicates_previous=duplicates_previous,
    )


def save_figures(fig: plt.Figure, stem: str, output_dir: Path, paper_plots_dir: Path, formats: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_plots_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        data_path = output_dir / f"{stem}.{fmt}"
        paper_path = paper_plots_dir / f"{stem}.{fmt}"
        fig.savefig(data_path, bbox_inches="tight", dpi=300)
        fig.savefig(paper_path, bbox_inches="tight", dpi=300)


def plot_trajectory(
    manifest: Sequence[CheckpointEntry],
    coords: np.ndarray,
    explained: np.ndarray,
    summary: PathSummary,
    output_dir: Path,
    paper_plots_dir: Path,
    formats: Sequence[str],
) -> None:
    labels = [entry.label for entry in manifest]
    steps = np.arange(len(manifest))
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=steps,
        cmap="viridis",
        s=86,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.scatter(coords[0, 0], coords[0, 1], marker="*", s=240, color="#E69F00", edgecolor="black", zorder=4)
    ax.scatter(coords[-1, 0], coords[-1, 1], marker="X", s=170, color="#D55E00", edgecolor="black", zorder=4)
    for i in range(1, len(manifest)):
        dx = coords[i, 0] - coords[i - 1, 0]
        dy = coords[i, 1] - coords[i - 1, 1]
        ax.annotate(
            "",
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(coords[i - 1, 0], coords[i - 1, 1]),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5, shrinkA=8, shrinkB=8),
            zorder=2,
        )
        if abs(dx) + abs(dy) < 1e-12:
            ax.text(coords[i, 0], coords[i, 1], "duplicate", fontsize=8, color="#D55E00")
    for i, label in enumerate(labels):
        offset_y = 0.015 * max(1.0, np.ptp(coords[:, 1]))
        ax.text(coords[i, 0], coords[i, 1] + offset_y, label, fontsize=9, ha="center", va="bottom")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Chronological checkpoint index")
    x_exp = explained[0] * 100 if len(explained) > 0 else 0.0
    y_exp = explained[1] * 100 if len(explained) > 1 else 0.0
    ax.set_xlabel(f"Manifold coordinate 1 ({x_exp:.1f}% variance)")
    ax.set_ylabel(f"Manifold coordinate 2 ({y_exp:.1f}% variance)")
    ax.set_title("Qwen3-VL-8B LoRA Weight-Update Manifold Trajectory")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    info = (
        f"Endpoint displacement: {summary.endpoint_displacement:.3g}\n"
        f"Path length: {summary.cumulative_path_length_nonduplicate:.3g}\n"
        f"Straightness: {summary.straightness_nonduplicate:.3f}"
    )
    ax.text(
        0.02,
        0.02,
        info,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )
    fig.tight_layout()
    save_figures(fig, "qwen3vl_lora_weight_manifold_trajectory", output_dir, paper_plots_dir, formats)
    plt.close(fig)


def plot_trajectory_3d(
    manifest: Sequence[CheckpointEntry],
    coords: np.ndarray,
    explained: np.ndarray,
    summary: PathSummary,
    output_dir: Path,
    paper_plots_dir: Path,
    formats: Sequence[str],
) -> None:
    """3D isometric view of the checkpoint manifold trajectory."""
    if coords.shape[1] < 3:
        print("  Skipping 3D plot: fewer than 3 embedding dimensions available.")
        return

    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

    labels = [entry.label for entry in manifest]
    steps = np.arange(len(manifest))
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw trajectory line segments coloured by progress
    for i in range(1, len(manifest)):
        progress = i / max(len(manifest) - 1, 1)
        color = plt.cm.viridis(progress)
        ax.plot(
            [x[i - 1], x[i]],
            [y[i - 1], y[i]],
            [z[i - 1], z[i]],
            color=color,
            lw=2.2,
            zorder=2,
        )

    # Draw 3D arrows along segments
    for i in range(1, len(manifest)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        dz = z[i] - z[i - 1]
        length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if length < 1e-12:
            continue
        progress = i / max(len(manifest) - 1, 1)
        color = plt.cm.viridis(progress)
        ax.quiver(
            x[i - 1], y[i - 1], z[i - 1],
            dx, dy, dz,
            arrow_length_ratio=min(0.15, 0.6 / max(length, 1e-6)),
            color=color,
            linewidth=1.6,
            zorder=3,
        )

    # Scatter all checkpoints
    scatter = ax.scatter(
        x, y, z,
        c=steps,
        cmap="viridis",
        s=90,
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        zorder=4,
    )

    # Mark base (star) and final (X)
    ax.scatter([x[0]], [y[0]], [z[0]], marker="*", s=280, color="#E69F00", edgecolor="black", zorder=5)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], marker="X", s=200, color="#D55E00", edgecolor="black", zorder=5)

    # Labels
    for i, label in enumerate(labels):
        ax.text(x[i], y[i], z[i], f"  {label}", fontsize=9, ha="left", va="bottom", zorder=6)

    # Axes
    x_exp = explained[0] * 100 if len(explained) > 0 else 0.0
    y_exp = explained[1] * 100 if len(explained) > 1 else 0.0
    z_exp = explained[2] * 100 if len(explained) > 2 else 0.0
    ax.set_xlabel(f"PC1 ({x_exp:.1f}%)", labelpad=10)
    ax.set_ylabel(f"PC2 ({y_exp:.1f}%)", labelpad=10)
    ax.set_zlabel(f"PC3 ({z_exp:.1f}%)", labelpad=10)
    ax.set_title("Qwen3-VL-8B LoRA Weight-Update Manifold (3D Isometric)")

    # Isometric-like viewing angle
    ax.view_init(elev=25, azim=135)
    ax.set_box_aspect([1, 1, 1])

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.12, shrink=0.65)
    cbar.set_label("Checkpoint index")

    # Info box
    info = (
        f"Displacement: {summary.endpoint_displacement:.3g}\n"
        f"Path length: {summary.cumulative_path_length_nonduplicate:.3g}\n"
        f"Straightness: {summary.straightness_nonduplicate:.3f}"
    )
    ax.text2D(
        0.02,
        0.02,
        info,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )

    fig.tight_layout()
    save_figures(fig, "qwen3vl_lora_weight_manifold_3d", output_dir, paper_plots_dir, formats)
    plt.close(fig)


def plot_manifold_surface_3d(
    manifest: Sequence[CheckpointEntry],
    coords: np.ndarray,
    explained: np.ndarray,
    summary: PathSummary,
    output_dir: Path,
    paper_plots_dir: Path,
    formats: Sequence[str],
) -> None:
    """Translucent triangulated surface through checkpoint coordinates.

    With one fine-tuning run, the observed object is a 1D trajectory sampled at
    checkpoints. This surface is therefore a visualization aid: a piecewise-linear
    sheet through the nonduplicate 3D kernel-PCA points, not a fitted generative
    model of the complete weight manifold.
    """
    if coords.shape[1] < 3:
        print("  Skipping 3D surface plot: fewer than 3 embedding dimensions available.")
        return

    import matplotlib.tri as mtri

    labels = [entry.label for entry in manifest]
    steps = np.arange(len(manifest))
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Remove exact duplicate coordinates before triangulation, keeping the
    # chronological plotted points unchanged for the trajectory overlay.
    unique_indices: List[int] = []
    seen = set()
    for idx, point in enumerate(coords[:, :3]):
        key = tuple(np.round(point, decimals=10))
        if key in seen:
            continue
        seen.add(key)
        unique_indices.append(idx)

    if len(unique_indices) < 3:
        print("  Skipping 3D surface plot: fewer than 3 unique points available.")
        return

    xu = x[unique_indices]
    yu = y[unique_indices]
    zu = z[unique_indices]

    try:
        triangulation = mtri.Triangulation(xu, yu)
    except Exception as exc:
        print(f"  Skipping 3D surface plot: triangulation failed ({exc}).")
        return

    fig = plt.figure(figsize=(10.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_trisurf(
        triangulation,
        zu,
        cmap="viridis",
        alpha=0.28,
        linewidth=0.45,
        edgecolor="#666666",
        antialiased=True,
        shade=True,
        zorder=1,
    )
    surface.set_clim(vmin=float(np.nanmin(z)), vmax=float(np.nanmax(z)))

    # Overlay chronological trajectory on top of the surface.
    for i in range(1, len(manifest)):
        progress = i / max(len(manifest) - 1, 1)
        color = plt.cm.plasma(progress)
        ax.plot(
            [x[i - 1], x[i]],
            [y[i - 1], y[i]],
            [z[i - 1], z[i]],
            color=color,
            lw=2.5,
            zorder=3,
        )
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        dz = z[i] - z[i - 1]
        length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if length >= 1e-12:
            ax.quiver(
                x[i - 1], y[i - 1], z[i - 1],
                dx, dy, dz,
                arrow_length_ratio=min(0.13, 0.55 / max(length, 1e-6)),
                color=color,
                linewidth=1.5,
                zorder=4,
            )

    scatter = ax.scatter(
        x,
        y,
        z,
        c=steps,
        cmap="plasma",
        s=92,
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        zorder=5,
    )
    ax.scatter([x[0]], [y[0]], [z[0]], marker="*", s=280, color="#E69F00", edgecolor="black", zorder=6)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], marker="X", s=200, color="#D55E00", edgecolor="black", zorder=6)

    for i, label in enumerate(labels):
        ax.text(x[i], y[i], z[i], f"  {label}", fontsize=9, ha="left", va="bottom", zorder=7)

    x_exp = explained[0] * 100 if len(explained) > 0 else 0.0
    y_exp = explained[1] * 100 if len(explained) > 1 else 0.0
    z_exp = explained[2] * 100 if len(explained) > 2 else 0.0
    ax.set_xlabel(f"PC1 ({x_exp:.1f}%)", labelpad=10)
    ax.set_ylabel(f"PC2 ({y_exp:.1f}%)", labelpad=10)
    ax.set_zlabel(f"PC3 ({z_exp:.1f}%)", labelpad=10)
    ax.set_title("Approximate Triangulated Sheet Through Qwen3-VL LoRA Checkpoints")
    ax.view_init(elev=25, azim=135)
    ax.set_box_aspect([1, 1, 0.75])

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.12, shrink=0.65)
    cbar.set_label("Checkpoint index")

    info = (
        "Triangulated visual sheet through checkpoints\n"
        "Not a learned 2D manifold\n"
        f"Displacement: {summary.endpoint_displacement:.3g}\n"
        f"Path length: {summary.cumulative_path_length_nonduplicate:.3g}"
    )
    ax.text2D(
        0.02,
        0.02,
        info,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="#CCCCCC"),
    )

    fig.tight_layout()
    save_figures(fig, "qwen3vl_lora_weight_manifold_surface_3d", output_dir, paper_plots_dir, formats)
    plt.close(fig)


def plot_smooth_weight_manifold_surface_3d(
    manifest: Sequence[CheckpointEntry],
    coords: np.ndarray,
    explained: np.ndarray,
    summary: PathSummary,
    output_dir: Path,
    paper_plots_dir: Path,
    formats: Sequence[str],
) -> None:
    """Smooth tube around the weight-derived checkpoint trajectory.

    The centerline is a cubic spline through the effective-LoRA kernel-PCA
    coordinates. Those coordinates are computed from checkpoint weights via the
    exact low-rank LoRA kernel, so this is a smoothed visualization of the
    observed weight-update trajectory rather than a separately identified 2D
    manifold surface.
    """
    if coords.shape[1] < 3:
        print("  Skipping smooth 3D surface plot: fewer than 3 embedding dimensions available.")
        return

    try:
        from scipy.interpolate import CubicSpline
    except Exception as exc:
        print(f"  Skipping smooth 3D surface plot: SciPy CubicSpline unavailable ({exc}).")
        return

    labels = [entry.label for entry in manifest]
    steps = np.arange(len(manifest))

    # Drop exact duplicate coordinates before spline fitting, but keep all
    # original points for the overlay. checkpoint-final duplicates checkpoint-348.
    path_indices: List[int] = [0]
    for idx in range(1, len(manifest)):
        if np.linalg.norm(coords[idx, :3] - coords[path_indices[-1], :3]) > 1e-10:
            path_indices.append(idx)

    if len(path_indices) < 4:
        print("  Skipping smooth 3D surface plot: at least four unique points are preferred for cubic splines.")
        return

    path = coords[path_indices, :3]
    chord = np.zeros(len(path), dtype=np.float64)
    chord[1:] = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    if chord[-1] <= 0:
        print("  Skipping smooth 3D surface plot: degenerate trajectory.")
        return
    t = chord / chord[-1]

    splines = [CubicSpline(t, path[:, dim], bc_type="natural") for dim in range(3)]
    t_dense = np.linspace(0.0, 1.0, 260)
    center = np.column_stack([spline(t_dense) for spline in splines])
    tangent = np.column_stack([spline(t_dense, 1) for spline in splines])
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(tangent_norm, 1e-12)

    # Parallel-transport-like frame: project the previous normal onto the new
    # tangent plane to avoid noisy Frenet flips from only a few checkpoints.
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal = np.zeros_like(tangent)
    binormal = np.zeros_like(tangent)
    n0 = up - np.dot(up, tangent[0]) * tangent[0]
    if np.linalg.norm(n0) < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        n0 = up - np.dot(up, tangent[0]) * tangent[0]
    normal[0] = n0 / max(np.linalg.norm(n0), 1e-12)
    binormal[0] = np.cross(tangent[0], normal[0])
    binormal[0] = binormal[0] / max(np.linalg.norm(binormal[0]), 1e-12)
    for idx in range(1, len(t_dense)):
        n = normal[idx - 1] - np.dot(normal[idx - 1], tangent[idx]) * tangent[idx]
        if np.linalg.norm(n) < 1e-8:
            n = up - np.dot(up, tangent[idx]) * tangent[idx]
        normal[idx] = n / max(np.linalg.norm(n), 1e-12)
        b = np.cross(tangent[idx], normal[idx])
        binormal[idx] = b / max(np.linalg.norm(b), 1e-12)

    coord_span = np.linalg.norm(np.ptp(coords[:, :3], axis=0))
    radius_major = 0.042 * coord_span
    radius_minor = 0.024 * coord_span
    angles = np.linspace(0.0, 2.0 * np.pi, 32)
    cos_a = np.cos(angles)[None, :]
    sin_a = np.sin(angles)[None, :]
    X = center[:, 0, None] + radius_major * normal[:, 0, None] * cos_a + radius_minor * binormal[:, 0, None] * sin_a
    Y = center[:, 1, None] + radius_major * normal[:, 1, None] * cos_a + radius_minor * binormal[:, 1, None] * sin_a
    Z = center[:, 2, None] + radius_major * normal[:, 2, None] * cos_a + radius_minor * binormal[:, 2, None] * sin_a

    facecolors = plt.cm.viridis(t_dense)[:, None, :]
    facecolors = np.repeat(facecolors, len(angles), axis=1)
    facecolors[..., 3] = 0.24

    fig = plt.figure(figsize=(10.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        zorder=1,
    )

    # Smooth centerline and original checkpoint-to-checkpoint arrows.
    ax.plot(center[:, 0], center[:, 1], center[:, 2], color="#222222", lw=2.6, alpha=0.88, zorder=3)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    for idx in range(1, len(manifest)):
        dx = x[idx] - x[idx - 1]
        dy = y[idx] - y[idx - 1]
        dz = z[idx] - z[idx - 1]
        length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if length < 1e-12:
            continue
        progress = idx / max(len(manifest) - 1, 1)
        color = plt.cm.plasma(progress)
        ax.quiver(
            x[idx - 1], y[idx - 1], z[idx - 1],
            dx, dy, dz,
            arrow_length_ratio=min(0.12, 0.5 / max(length, 1e-6)),
            color=color,
            linewidth=1.35,
            zorder=4,
        )

    scatter = ax.scatter(
        x,
        y,
        z,
        c=steps,
        cmap="plasma",
        s=92,
        edgecolor="black",
        linewidth=0.8,
        depthshade=False,
        zorder=5,
    )
    ax.scatter([x[0]], [y[0]], [z[0]], marker="*", s=285, color="#E69F00", edgecolor="black", zorder=6)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], marker="X", s=205, color="#D55E00", edgecolor="black", zorder=6)

    for idx, label in enumerate(labels):
        ax.text(x[idx], y[idx], z[idx], f"  {label}", fontsize=9, ha="left", va="bottom", zorder=7)

    x_exp = explained[0] * 100 if len(explained) > 0 else 0.0
    y_exp = explained[1] * 100 if len(explained) > 1 else 0.0
    z_exp = explained[2] * 100 if len(explained) > 2 else 0.0
    ax.set_xlabel(f"PC1 ({x_exp:.1f}%)", labelpad=10)
    ax.set_ylabel(f"PC2 ({y_exp:.1f}%)", labelpad=10)
    ax.set_zlabel(f"PC3 ({z_exp:.1f}%)", labelpad=10)
    ax.set_title("Smoothed Qwen3-VL LoRA Weight-Update Trajectory Tube")
    ax.view_init(elev=25, azim=135)
    ax.set_box_aspect([1, 1, 0.78])

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.12, shrink=0.65)
    cbar.set_label("Checkpoint index")

    info = (
        "Spline tube around checkpoint trajectory\n"
        "Not a separately estimated 2D manifold\n"
        f"Displacement: {summary.endpoint_displacement:.3g}\n"
        f"Path length: {summary.cumulative_path_length_nonduplicate:.3g}"
    )
    ax.text2D(
        0.02,
        0.02,
        info,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.86, edgecolor="#CCCCCC"),
    )

    fig.tight_layout()
    save_figures(fig, "qwen3vl_lora_weight_update_smooth_trajectory_tube_3d", output_dir, paper_plots_dir, formats)
    plt.close(fig)


def plot_radial_path(
    point_records: Sequence[Dict[str, Any]],
    output_dir: Path,
    paper_plots_dir: Path,
    formats: Sequence[str],
) -> None:
    x = np.arange(len(point_records))
    labels = [str(record["label"]) for record in point_records]
    norms = np.asarray([float(record["radial_norm"]) for record in point_records])
    cumulative = np.asarray([float(record["cumulative_path_length"]) for record in point_records])
    loss = np.asarray([
        np.nan if record.get("loss") is None else float(record["loss"]) for record in point_records
    ])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), sharex=True)
    axes[0].plot(x, norms, marker="o", color="#0072B2", lw=2)
    axes[0].set_ylabel(r"$||\Delta W_t||_F$")
    axes[0].set_title("Growth of the Effective LoRA Update During Fine-Tuning")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, cumulative, marker="o", color="#009E73", lw=2)
    axes[1].set_ylabel("Cumulative path length")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, loss, marker="o", color="#D55E00", lw=2)
    axes[2].set_ylabel("Training loss")
    axes[2].set_xlabel("Checkpoint")
    axes[2].grid(True, alpha=0.25)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    save_figures(fig, "qwen3vl_lora_radial_path_loss", output_dir, paper_plots_dir, formats)
    plt.close(fig)


def plot_segments(
    segment_records: Sequence[Dict[str, Any]],
    output_dir: Path,
    paper_plots_dir: Path,
    formats: Sequence[str],
) -> None:
    x = np.arange(len(segment_records))
    labels = [str(record["segment_index"]) + ": " + str(record["to_label"]) for record in segment_records]
    distances = np.asarray([float(record["distance"]) for record in segment_records])
    angles = np.asarray(
        [
            np.nan
            if record.get("turn_angle_degrees_with_previous_segment") is None
            else float(record["turn_angle_degrees_with_previous_segment"])
            for record in segment_records
        ]
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
    axes[0].bar(x, distances, color="#56B4E9", edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Step displacement")
    axes[0].set_title("Local Movement and Curvature of the Fine-Tuning Trajectory")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(x, angles, marker="o", color="#CC79A7", lw=2)
    axes[1].set_ylabel("Turning angle (degrees)")
    axes[1].set_xlabel("Trajectory segment")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    save_figures(fig, "qwen3vl_lora_segment_curvature", output_dir, paper_plots_dir, formats)
    plt.close(fig)


def plot_module_heatmap(
    module_step_records: Sequence[Dict[str, Any]],
    output_dir: Path,
    paper_plots_dir: Path,
    formats: Sequence[str],
) -> None:
    groups = sorted({str(record["module_type"]) for record in module_step_records})
    segments = []
    for record in module_step_records:
        segment = str(record["segment"])
        if segment not in segments:
            segments.append(segment)
    matrix = np.zeros((len(groups), len(segments)), dtype=np.float64)
    group_to_idx = {group: idx for idx, group in enumerate(groups)}
    segment_to_idx = {segment: idx for idx, segment in enumerate(segments)}
    for record in module_step_records:
        matrix[group_to_idx[str(record["module_type"])]][segment_to_idx[str(record["segment"])]] = float(
            record["movement_norm"]
        )

    fig, ax = plt.subplots(figsize=(10, 5.4))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xticks(np.arange(len(segments)))
    ax.set_xticklabels(segments, rotation=35, ha="right")
    ax.set_xlabel("Consecutive checkpoint segment")
    ax.set_ylabel("LoRA target module")
    ax.set_title("Where the Effective LoRA Update Moves During Fine-Tuning")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Segment movement norm")
    fig.tight_layout()
    save_figures(fig, "qwen3vl_lora_module_movement_heatmap", output_dir, paper_plots_dir, formats)
    plt.close(fig)


def write_methods_paragraph(path: Path, summary: PathSummary, validation: ValidationReport) -> None:
    duplicate_distance = (
        f"{validation.duplicate_previous_distance:.6g}"
        if validation.duplicate_previous_distance is not None
        else "not available"
    )
    paragraph = r"""# Manuscript-ready methods note: checkpoint manifold trajectory

We analyze fine-tuning dynamics by viewing each saved Qwen3-VL-8B LoRA checkpoint as a point in the space of effective model updates. For every LoRA-adapted layer $\ell$, the checkpoint-specific update is defined as

$$
\Delta W_\ell(t)=\frac{\alpha}{r}B_\ell(t)A_\ell(t),
$$

where $A_\ell$ and $B_\ell$ are the learned LoRA factors, $r$ is the adapter rank, and $\alpha$ is the LoRA scaling parameter. We use $\Delta W_\ell$ rather than the raw LoRA factors because the product $B_\ell A_\ell$ is the actual induced change to the frozen pretrained weights and is invariant to reciprocal rescalings of the low-rank factors. The pretrained model is included as the origin, $\Delta W=0$.

Pairwise checkpoint geometry is computed exactly without materializing the dense 8B model or the dense LoRA products. For checkpoints $i$ and $j$, the Frobenius inner product of effective updates is

$$
\langle \Delta_i,\Delta_j\rangle
=\sum_\ell s_i s_j\operatorname{{tr}}\left[(B_{\ell,i}^{\top}B_{\ell,j})(A_{\ell,j}A_{\ell,i}^{\top})\right],\quad s=\alpha/r.
$$

This yields a positive-semidefinite kernel over checkpoints, from which Euclidean distances, angular distances between nonzero update directions, and low-dimensional kernel-PCA/MDS coordinates are derived. Chronological arrows between embedded checkpoints define the fine-tuning trajectory. We summarize this path using radial update norm, per-step displacement, cumulative path length, endpoint displacement, straightness ratio, turning angle, and module-wise movement contributions.
"""
    paragraph += (
        f"\nIn this run, the endpoint displacement is {summary.endpoint_displacement:.6g}, "
        f"the nonduplicate path length is {summary.cumulative_path_length_nonduplicate:.6g}, "
        f"and the nonduplicate straightness ratio is {summary.straightness_nonduplicate:.6g}. "
        f"The final checkpoint duplicate distance to its predecessor is {duplicate_distance}.\n"
    )
    paragraph += r"""

This analysis uses the same manifold viewpoint as density-based representation methods, but it does not fit a flow model because the checkpoint sample size is small. Instead, the trajectory is obtained from the exact Riemannian/kernel geometry of LoRA-induced model updates.
"""
    path.write_text(paragraph, encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    paper_plots_dir = args.paper_plots_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dirs = discover_checkpoint_dirs(checkpoint_root)
    include_base = not args.no_base
    manifest = build_manifest(checkpoint_dirs, include_base=include_base)
    modules = validate_adapter_key_shapes(manifest)
    print(f"Discovered {len(checkpoint_dirs)} adapter checkpoints and {len(modules)} LoRA modules.")

    kernel, module_kernels = compute_low_rank_kernel(manifest, modules)
    squared_distances = squared_distances_from_kernel(kernel)
    distances = np.sqrt(squared_distances)
    angular_distances = angular_distances_from_kernel(kernel)

    coords, evals, explained = coordinates_from_kernel(kernel, dims=args.dims)
    spherical_coords, spherical_evals, spherical_explained, spherical_indices = spherical_mds_coordinates(
        angular_distances, manifest, dims=args.dims
    )

    summary, point_records, segment_records = compute_trajectory_metrics(
        manifest, kernel, distances, duplicate_tolerance=args.duplicate_tolerance
    )
    module_final_records = grouped_norm_records(manifest, module_kernels, group="module_type")
    layer_final_records = grouped_norm_records(manifest, module_kernels, group="layer")
    block_final_records = grouped_norm_records(manifest, module_kernels, group="block_type")
    module_step_records = grouped_step_movement_records(manifest, module_kernels, group="module_type")
    layer_step_records = grouped_step_movement_records(manifest, module_kernels, group="layer")

    validation = build_validation_report(
        kernel,
        distances,
        manifest,
        modules,
        module_kernels,
        duplicate_tolerance=args.duplicate_tolerance,
        skip_materialization=args.skip_validation_materialization,
    )

    labels = [entry.label for entry in manifest]
    manifest_records = [asdict(entry) for entry in manifest]
    write_json(output_dir / "manifest.json", manifest_records)
    records_to_csv(output_dir / "manifest.csv", manifest_records)

    np.save(output_dir / "effective_lora_kernel.npy", kernel)
    np.save(output_dir / "euclidean_distances.npy", distances)
    np.save(output_dir / "angular_distances.npy", angular_distances)
    matrix_to_csv(output_dir / "effective_lora_kernel.csv", kernel, labels)
    matrix_to_csv(output_dir / "euclidean_distances.csv", distances, labels)
    matrix_to_csv(output_dir / "angular_distances.csv", angular_distances, labels)

    coord_records = coordinate_records(manifest, coords, prefix="kpca_dim", explained=explained)
    spherical_records = coordinate_records(manifest, spherical_coords, prefix="spherical_dim", explained=spherical_explained)
    records_to_csv(output_dir / "coordinates_kernel_pca.csv", coord_records)
    records_to_csv(output_dir / "coordinates_spherical_mds.csv", spherical_records)
    write_json(
        output_dir / "coordinates.json",
        {
            "kernel_pca": coord_records,
            "kernel_pca_eigenvalues": evals,
            "kernel_pca_explained_variance": explained,
            "spherical_mds": spherical_records,
            "spherical_mds_eigenvalues": spherical_evals,
            "spherical_mds_explained_variance": spherical_explained,
            "spherical_mds_nonbase_indices": spherical_indices,
        },
    )

    records_to_csv(output_dir / "trajectory_point_metrics.csv", point_records)
    records_to_csv(output_dir / "trajectory_segment_metrics.csv", segment_records)
    records_to_csv(output_dir / "module_type_final_contributions.csv", module_final_records)
    records_to_csv(output_dir / "layer_final_contributions.csv", layer_final_records)
    records_to_csv(output_dir / "block_type_final_contributions.csv", block_final_records)
    records_to_csv(output_dir / "module_type_step_movements.csv", module_step_records)
    records_to_csv(output_dir / "layer_step_movements.csv", layer_step_records)

    write_json(
        output_dir / "trajectory_summary.json",
        {
            "path_summary": asdict(summary),
            "validation": asdict(validation),
            "kernel_pca_explained_variance": explained,
            "spherical_mds_explained_variance": spherical_explained,
            "checkpoint_root": str(checkpoint_root),
            "output_dir": str(output_dir),
            "paper_plots_dir": str(paper_plots_dir),
        },
    )

    plot_trajectory(manifest, coords, explained, summary, output_dir, paper_plots_dir, args.formats)
    plot_trajectory_3d(manifest, coords, explained, summary, output_dir, paper_plots_dir, args.formats)
    plot_manifold_surface_3d(manifest, coords, explained, summary, output_dir, paper_plots_dir, args.formats)
    plot_smooth_weight_manifold_surface_3d(manifest, coords, explained, summary, output_dir, paper_plots_dir, args.formats)
    plot_radial_path(point_records, output_dir, paper_plots_dir, args.formats)
    plot_segments(segment_records, output_dir, paper_plots_dir, args.formats)
    plot_module_heatmap(module_step_records, output_dir, paper_plots_dir, args.formats)
    write_methods_paragraph(output_dir / "manuscript_methods_note.md", summary, validation)

    print("\nDone. Key outputs:")
    print(f"  Data directory: {output_dir}")
    print(f"  Paper figures:  {paper_plots_dir}")
    print(f"  Endpoint displacement: {summary.endpoint_displacement:.6g}")
    print(f"  Nonduplicate path length: {summary.cumulative_path_length_nonduplicate:.6g}")
    print(f"  Nonduplicate straightness: {summary.straightness_nonduplicate:.6g}")
    if validation.duplicate_previous_distance is not None:
        print(f"  Final-to-previous distance: {validation.duplicate_previous_distance:.6g}")
    print(f"  Kernel min eigenvalue: {validation.kernel_min_eigenvalue:.6g}")


if __name__ == "__main__":
    main()
