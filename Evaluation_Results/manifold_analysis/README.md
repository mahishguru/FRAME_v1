# Multi-Run LoRA Weight-Manifold Analysis

This folder contains the LoRA checkpoint manifold analysis used for the Qwen3-VL and Pixtral fine-tuning trajectories. The current workflow has two stages:

1. `multi_run_manifold.py` computes and saves the weight-space geometry from all checkpoints.
2. `plot_manifold.py` loads the saved data and renders paper-style figures quickly.

Both scripts support presets:

```bash
# Qwen3-VL, default
/data/mguru/04_Finetuning/finetune/bin/python \
  fine_tune_llm_post_processing/evaluation/manifold_analysis/multi_run_manifold.py \
  --preset qwen --skip-plots

/data/mguru/04_Finetuning/finetune/bin/python \
  fine_tune_llm_post_processing/evaluation/manifold_analysis/plot_manifold.py \
  --preset qwen --only-3d

# Pixtral
/data/mguru/04_Finetuning/finetune/bin/python \
  fine_tune_llm_post_processing/evaluation/manifold_analysis/multi_run_manifold.py \
  --preset pixtral --skip-plots

/data/mguru/04_Finetuning/finetune/bin/python \
  fine_tune_llm_post_processing/evaluation/manifold_analysis/plot_manifold.py \
  --preset pixtral --only-3d
```

## Method Summary

Each LoRA checkpoint is represented as a point in the space of effective LoRA weight updates. For a LoRA module `ell`, the effective update is

```text
Delta W_ell(t) = (alpha / r) * B_ell(t) @ A_ell(t)
```

The pretrained base model is represented as the zero-update point. For every pair of checkpoints, the analysis computes an exact Frobenius inner-product kernel over effective LoRA updates:

```text
K_ij = <Delta W_i, Delta W_j>_F
```

The implementation avoids materializing dense update matrices by using the low-rank LoRA identity module-by-module:

```text
<B_i A_i, B_j A_j>_F = sum((B_i.T @ B_j) * (A_j @ A_i.T))
```

The diagonal of the kernel gives the squared update norm:

```text
||Delta W_i||_F = sqrt(K_ii)
```

After adding the base model as the zero-update row/column, the centered kernel is embedded with kernel PCA. The first two kernel-PCA coordinates define the horizontal axes:

```text
x = PC1 of LoRA-update geometry
y = PC2 of LoRA-update geometry
z = ||Delta W||_F
```

The 3D surface is an RBF interpolation of LoRA update norm over the 2D kernel-PCA coordinates:

```text
z = f(PC1, PC2) ~= ||Delta W||_F
```

The surface is fitted from all available checkpoints across the 50%, 60%, 70%, 80%, and 100% runs. The red curve highlights only the `100% (set2)` trajectory:

```text
base -> checkpoint-50 -> checkpoint-100 -> ... -> final
```

The red checkpoint markers are the actual checkpoint positions. The red curve is a smoothed visual trajectory through these points, with a monotone interpolated height profile so the path climbs smoothly from the base instead of inheriting clipped RBF artifacts near the origin.

## Current Results

Qwen3-VL:

- 32 checkpoints from 5 runs.
- 252 LoRA modules.
- PC1 explains 34.4%; PC2 explains 23.7%.
- Data: `multi_run_manifold_results/`.
- Paper plots: `../plots_paper/qwen3vl_manifold/`.
- Main 3D figure: `qwen3vl_multi_run_weight_manifold_3d.png/pdf`.

Pixtral:

- 32 checkpoints from 5 runs.
- 448 LoRA modules.
- PC1 explains 33.0%; PC2 explains 24.3%.
- Data: `pixtral_multi_run_manifold_results/`.
- Paper plots: `../plots_paper/pixtral_manifold/`.
- Main 3D figure: `pixtral_multi_run_weight_manifold_3d.png/pdf`.

Saved manifold data products include:

- `multi_run_kernel.npy`: kernel over base plus adapter checkpoints.
- `multi_run_coords_2d.npy`: base plus checkpoint KPCA coordinates.
- `multi_run_explained.npy`: explained variance for PC1/PC2.
- `multi_run_base_coord.npy`: base coordinate.
- `multi_run_adapter_coords.npy`: checkpoint PC1/PC2 coordinates.
- `multi_run_adapter_norms.npy`: checkpoint `||Delta W||_F` values.
- `multi_run_manifest.json`: coordinate and metadata table.
- `multi_run_points.json`: checkpoint metadata used by the plotter.

## Cross-Architecture Interpretation

Qwen and Pixtral have different architectures, so their LoRA weight spaces are not directly comparable point-by-point. The two figures are not plotted in a shared coordinate system. Instead, each architecture gets its own independent kernel and kernel-PCA embedding:

```text
Qwen checkpoints    -> Qwen LoRA kernel    -> Qwen PC1/PC2
Pixtral checkpoints -> Pixtral LoRA kernel -> Pixtral PC1/PC2
```

Therefore, `PC1` in Qwen is not the same physical direction as `PC1` in Pixtral. The figures can still look similar because they use the same analysis procedure, the same plotting convention, and similar checkpoint schedules. In both cases, LoRA fine-tuning tends to increase the effective update norm over training, so the highlighted trajectory naturally climbs away from the base.

Recommended paper wording:

```text
For each architecture, we compute an independent kernel-PCA embedding of effective LoRA weight updates. The resulting manifolds are architecture-specific and are not placed in a shared coordinate system. They should be interpreted as parallel within-model trajectory analyses under a common geometric procedure and plotting convention.
```

## Paper-Ready Method Note

For each fine-tuned model, we represent every LoRA checkpoint by its effective low-rank weight update, `Delta W = (alpha/r)BA`, across all adapted modules. We compute the exact pairwise Frobenius inner-product kernel between checkpoints using low-rank identities, avoiding dense reconstruction of the full model updates. The pretrained base model is included as the zero-update point. Kernel PCA on the centered kernel provides the first two manifold coordinates, while the vertical coordinate is the Frobenius norm of the LoRA update, `||Delta W||_F`. A smooth RBF surface is fitted from checkpoints across all data-percentage runs, and the 100% fine-tuning trajectory is highlighted as a temporal path from the base checkpoint to the final adapter. Because Qwen and Pixtral have different architectures, their embeddings are computed independently and are compared qualitatively rather than as coordinates in a shared weight space.
