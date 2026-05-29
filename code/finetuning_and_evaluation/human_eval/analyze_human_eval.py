#!/usr/bin/env python3
"""Analyze FRAME benchmark human-evaluation ratings.

The script normalizes expert ratings, validates DOI coverage against the
benchmark JSON files, and creates three manuscript-ready figures plus a
summary table for human validation of FRAME input/output extractions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:  # pragma: no cover - fallback handles missing sklearn
    cohen_kappa_score = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_INPUT = PROJECT_ROOT / "final_input.json"
DEFAULT_FINAL_OUTPUT = PROJECT_ROOT / "final_output.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

CRITERIA = ["Correctness", "Completeness", "Coherence", "Faithfulness"]
CRITERION_ALIASES = {
    "Correctness": {"correctness", "correct", "accuracy", "accurate"},
    "Completeness": {"completeness", "complete", "coverage", "comprehensive"},
    "Coherence": {"coherence", "coherent", "clarity", "readability", "structure"},
    "Faithfulness": {"faithfulness", "faithful", "factuality", "hallucination"},
}
ARTIFACT_ORDER = ["input", "output", "combined"]
ARTIFACT_LABELS = {
    "input": "Input-key extraction",
    "output": "Output-key extraction",
    "combined": "Overall input/output extraction",
}
ARTIFACT_SHORT_LABELS = {
    "input": "Input keys",
    "output": "Output keys",
    "combined": "Input+output",
}
LIKERT_COLORS = {
    1: "#b2182b",
    2: "#ef8a62",
    3: "#e0e0e0",
    4: "#67a9cf",
    5: "#2166ac",
}
HEATMAP_COLORS = ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"]
UNIT_COLUMNS = ["doi_norm", "artifact_type", "criterion"]


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_doi(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or None


def looks_like_doi(value: Any) -> bool:
    normalized = normalize_doi(value)
    return bool(normalized and normalized.startswith("10_"))


def split_doi_title(value: Any) -> tuple[str | None, str | None, str | None]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None, None
    text = str(value).strip()
    if not text:
        return None, None, None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, None, None

    doi_line_index = None
    doi_raw = None
    for line_index, line in enumerate(lines):
        if looks_like_doi(line):
            doi_line_index = line_index
            doi_raw = line
            break

    if doi_raw is None:
        match = re.search(r"10[._/][A-Za-z0-9._/\-]+", text)
        if match:
            doi_raw = match.group(0)
            doi_line_index = 0

    if doi_raw is None:
        return None, None, text

    title_lines = [line for line_index, line in enumerate(lines) if line_index != doi_line_index]
    title = " ".join(title_lines).strip() or None
    return normalize_doi(doi_raw), doi_raw, title


def normalize_artifact(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = slugify(value)
    if not text:
        return None
    if "output" in text or text in {"out", "response", "answer", "strategy"}:
        return "output"
    if "input" in text or text in {"in", "prompt", "problem", "metadata"}:
        return "input"
    if "combined" in text or "overall" in text:
        return "combined"
    return text


def infer_artifact(default_artifact: str | None, *parts: Any) -> str:
    for part in parts:
        artifact = normalize_artifact(part)
        if artifact in {"input", "output", "combined"}:
            return artifact
    return default_artifact or "combined"


def normalize_criterion(value: Any) -> str | None:
    slug = slugify(value)
    if not slug:
        return None
    tokens = set(slug.split("_"))
    for criterion, aliases in CRITERION_ALIASES.items():
        if slug in aliases or tokens.intersection(aliases):
            return criterion
    return None


def parse_score(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        score = int(value)
    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            return None
        score = int(value)
    else:
        match = re.search(r"[1-5]", str(value))
        if not match:
            return None
        score = int(match.group(0))
    if 1 <= score <= 5:
        return score
    return None


def load_benchmark_keys(final_input: Path, final_output: Path) -> set[str]:
    keys: set[str] = set()
    for json_path in [final_input, final_output]:
        if not json_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        keys.update(normalize_doi(key) for key in payload.keys())
    return {key for key in keys if key}


def read_spreadsheet(path: Path) -> list[tuple[str, pd.DataFrame]]:
    suffix = path.suffix.lower()
    if suffix in {"", ".csv"}:
        return [(path.stem, pd.read_csv(path))]
    if suffix == ".tsv":
        return [(path.stem, pd.read_csv(path, sep="\t"))]
    if suffix in {".xlsx", ".xls"}:
        try:
            workbook = pd.ExcelFile(path)
        except ImportError as import_error:
            raise RuntimeError(
                f"Reading {path.name} requires openpyxl. Install it with `pip install openpyxl` "
                "or export the workbook sheets to CSV."
            ) from import_error
        sheets = []
        for sheet_name in workbook.sheet_names:
            sheets.append((sheet_name, pd.read_excel(workbook, sheet_name=sheet_name)))
        return sheets
    raise ValueError(f"Unsupported ratings file type: {path}")


def find_column(canonical_columns: dict[str, str], aliases: set[str]) -> str | None:
    for alias in aliases:
        if alias in canonical_columns:
            return canonical_columns[alias]
    for canonical_name, original_name in canonical_columns.items():
        if canonical_name in aliases:
            return original_name
        if any(alias in canonical_name.split("_") for alias in aliases):
            return original_name
    return None


def find_doi_column(dataframe: pd.DataFrame, canonical_columns: dict[str, str]) -> str | None:
    doi_column = find_column(
        canonical_columns,
        {"doi", "doi_norm", "normalized_doi", "normalised_doi", "paper_id", "case_id", "input_case"},
    )
    if doi_column:
        return doi_column

    best_column = None
    best_count = 0
    for column_name in dataframe.columns:
        sample_values = dataframe[column_name].dropna().head(30)
        doi_count = sum(looks_like_doi(value) for value in sample_values)
        if doi_count > best_count:
            best_column = column_name
            best_count = doi_count
    return best_column if best_count > 0 else None


def find_score_columns(dataframe: pd.DataFrame, reserved_columns: set[str]) -> list[tuple[str, str, str | None]]:
    score_columns = []
    for column_name in dataframe.columns:
        if column_name in reserved_columns:
            continue
        column_slug = slugify(column_name)
        criterion = normalize_criterion(column_slug)
        if criterion is None:
            continue
        artifact_hint = None
        if "input" in column_slug:
            artifact_hint = "input"
        elif "output" in column_slug:
            artifact_hint = "output"
        score_columns.append((column_name, criterion, artifact_hint))
    return score_columns


def infer_expert_id(path: Path, sheet_name: str, expert_id: str | None) -> str:
    if expert_id:
        return expert_id
    stem = slugify(path.stem)
    sheet_slug = slugify(sheet_name)
    if sheet_slug and sheet_slug not in stem and not any(token in sheet_slug for token in ["input", "output"]):
        return f"{stem}_{sheet_slug}"
    return stem


def row_title(row: pd.Series, title_column: str | None, embedded_title: str | None) -> str | None:
    if title_column is None:
        return embedded_title
    value = row.get(title_column)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return embedded_title
    title = str(value).strip()
    return title or embedded_title


def normalize_dataframe(
    dataframe: pd.DataFrame,
    source_path: Path,
    sheet_name: str,
    default_artifact: str | None,
    expert_id: str | None,
) -> pd.DataFrame:
    dataframe = dataframe.dropna(how="all").dropna(axis=1, how="all").copy()
    if dataframe.empty:
        return pd.DataFrame()

    canonical_columns = {slugify(column_name): column_name for column_name in dataframe.columns}
    doi_column = find_doi_column(dataframe, canonical_columns)
    if doi_column is None:
        raise ValueError(f"Could not find a DOI column in {source_path.name} / {sheet_name}")

    title_column = find_column(canonical_columns, {"title", "paper_title", "article_title"})
    expert_column = find_column(canonical_columns, {"expert_id", "expert", "rater", "rater_id", "domain_expert"})
    artifact_column = find_column(canonical_columns, {"artifact_type", "artifact", "file", "file_type", "rating_target"})
    criterion_column = find_column(canonical_columns, {"criterion", "criteria", "metric", "dimension"})
    rating_column = find_column(canonical_columns, {"score", "rating", "likert", "value"})

    reserved_columns = {column for column in [doi_column, title_column, expert_column, artifact_column, criterion_column, rating_column] if column}
    score_columns = find_score_columns(dataframe, reserved_columns)
    inferred_expert = infer_expert_id(source_path, sheet_name, expert_id)
    records: list[dict[str, Any]] = []

    for row_number, row in dataframe.iterrows():
        doi_norm, doi_raw, embedded_title = split_doi_title(row.get(doi_column))
        if doi_norm is None:
            continue

        title = row_title(row, title_column, embedded_title)
        row_expert = row.get(expert_column) if expert_column else inferred_expert
        row_expert_text = str(row_expert).strip() if row_expert is not None else inferred_expert
        row_artifact = row.get(artifact_column) if artifact_column else None
        base_artifact = infer_artifact(default_artifact, row_artifact, sheet_name, source_path.stem)

        if criterion_column and rating_column:
            criterion = normalize_criterion(row.get(criterion_column))
            score = parse_score(row.get(rating_column))
            if criterion and score is not None:
                records.append(
                    {
                        "expert_id": row_expert_text,
                        "source_file": str(source_path),
                        "source_sheet": sheet_name,
                        "row_number": int(row_number) + 2,
                        "doi_raw": doi_raw,
                        "doi_norm": doi_norm,
                        "title": title,
                        "artifact_type": base_artifact,
                        "criterion": criterion,
                        "score": score,
                    }
                )
            continue

        for score_column, criterion, artifact_hint in score_columns:
            score = parse_score(row.get(score_column))
            if score is None:
                continue
            artifact = artifact_hint or base_artifact
            records.append(
                {
                    "expert_id": row_expert_text,
                    "source_file": str(source_path),
                    "source_sheet": sheet_name,
                    "row_number": int(row_number) + 2,
                    "doi_raw": doi_raw,
                    "doi_norm": doi_norm,
                    "title": title,
                    "artifact_type": artifact,
                    "criterion": criterion,
                    "score": score,
                }
            )

    return pd.DataFrame.from_records(records)


def load_ratings(paths: list[Path], default_artifact: str | None, expert_id: str | None) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for sheet_name, dataframe in read_spreadsheet(path):
            normalized = normalize_dataframe(dataframe, path, sheet_name, default_artifact, expert_id)
            if not normalized.empty:
                frames.append(normalized)
    if not frames:
        raise ValueError("No valid human-evaluation ratings were found.")
    ratings = pd.concat(frames, ignore_index=True)
    ratings["score"] = ratings["score"].astype(int)
    ratings["artifact_type"] = ratings["artifact_type"].fillna("combined").map(normalize_artifact).fillna("combined")
    ratings["criterion"] = pd.Categorical(ratings["criterion"], categories=CRITERIA, ordered=True)
    return ratings.sort_values(["doi_norm", "artifact_type", "criterion", "expert_id"]).reset_index(drop=True)


def sort_artifacts(artifacts: list[str]) -> list[str]:
    return sorted(artifacts, key=lambda artifact: (ARTIFACT_ORDER.index(artifact) if artifact in ARTIFACT_ORDER else 99, artifact))


def display_artifact(artifact: str) -> str:
    return ARTIFACT_LABELS.get(artifact, artifact.replace("_", " ").title())


def display_artifact_short(artifact: str) -> str:
    return ARTIFACT_SHORT_LABELS.get(artifact, display_artifact(artifact))


def make_pivot(ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame()
    grouped = (
        ratings.groupby(UNIT_COLUMNS + ["expert_id"], observed=True)["score"]
        .mean()
        .round()
        .astype(int)
        .reset_index()
    )
    return grouped.pivot_table(index=UNIT_COLUMNS, columns="expert_id", values="score", aggfunc="mean", observed=True)


def quadratic_weighted_kappa(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if len(scores_a) < 2 or len(scores_b) < 2:
        return np.nan
    combined_values = list(scores_a) + list(scores_b)
    if len(set(combined_values)) == 1:
        return 1.0
    if cohen_kappa_score is None:
        return manual_quadratic_weighted_kappa(scores_a, scores_b)
    try:
        return float(cohen_kappa_score(scores_a, scores_b, labels=[1, 2, 3, 4, 5], weights="quadratic"))
    except Exception:
        return np.nan


def manual_quadratic_weighted_kappa(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    labels = np.array([1, 2, 3, 4, 5])
    observed = np.zeros((5, 5), dtype=float)
    for score_a, score_b in zip(scores_a, scores_b):
        observed[int(score_a) - 1, int(score_b) - 1] += 1
    hist_a = observed.sum(axis=1)
    hist_b = observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b) / max(observed.sum(), 1)
    weights = np.zeros((5, 5), dtype=float)
    for row_index, label_a in enumerate(labels):
        for column_index, label_b in enumerate(labels):
            weights[row_index, column_index] = ((label_a - label_b) ** 2) / 16.0
    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum()
    if denominator == 0:
        return 1.0 if numerator == 0 else np.nan
    return 1.0 - numerator / denominator


def gwet_ac2_quadratic(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Gwet's AC2 with quadratic weights for two raters on a 1-5 scale.

    Unlike QWK, AC2 does not suffer from the 'kappa paradox' when ratings
    are concentrated at the high end of the scale (Gwet, 2008; 2014).
    """
    q = 5
    N = len(scores_a)
    if N < 2:
        return np.nan

    conf = np.zeros((q, q), dtype=float)
    for sa, sb in zip(scores_a, scores_b):
        conf[int(sa) - 1, int(sb) - 1] += 1

    # Quadratic agreement weights: w(i,j) = 1 - (i-j)^2 / (q-1)^2
    max_diff_sq = float((q - 1) ** 2)
    w = np.zeros((q, q), dtype=float)
    for i in range(q):
        for j in range(q):
            w[i, j] = 1.0 - (i - j) ** 2 / max_diff_sq

    # Observed weighted agreement
    Pa = float(np.sum(w * conf)) / N

    # Average marginal probability per category
    pi = np.zeros(q, dtype=float)
    for k in range(q):
        pi[k] = (conf[k, :].sum() + conf[:, k].sum()) / (2.0 * N)

    # Tw_k = mean agreement weight to all other categories
    Tw = np.zeros(q, dtype=float)
    for k in range(q):
        Tw[k] = sum(w[k, l] for l in range(q) if l != k) / (q - 1)

    # Expected chance agreement (Gwet's model)
    Pe = float(sum(Tw[k] * pi[k] * (1.0 - pi[k]) for k in range(q)))

    if abs(1.0 - Pe) < 1e-12:
        return 1.0 if abs(1.0 - Pa) < 1e-12 else np.nan
    return (Pa - Pe) / (1.0 - Pe)


def krippendorff_alpha_interval(pivot: pd.DataFrame) -> float:
    if pivot.empty:
        return np.nan
    observed_diffs = []
    all_scores = []
    for _, row in pivot.iterrows():
        values = row.dropna().astype(float).to_numpy()
        all_scores.extend(values.tolist())
        if len(values) < 2:
            continue
        for score_a, score_b in itertools.combinations(values, 2):
            observed_diffs.append((score_a - score_b) ** 2)
    if not observed_diffs or len(all_scores) < 2:
        return np.nan
    observed_disagreement = float(np.mean(observed_diffs))
    pooled_scores = np.array(all_scores, dtype=float)
    expected_disagreement = 2.0 * len(pooled_scores) / (len(pooled_scores) - 1) * float(np.var(pooled_scores, ddof=0))
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else np.nan
    return 1.0 - observed_disagreement / expected_disagreement


def agreement_stats(ratings: pd.DataFrame) -> dict[str, float]:
    pivot = make_pivot(ratings)
    if pivot.empty:
        return {
            "agreement_units": 0,
            "exact_agreement": np.nan,
            "adjacent_agreement": np.nan,
            "mean_pairwise_qwk": np.nan,
            "krippendorff_alpha": np.nan,
        }

    complete_rows = pivot.dropna(how="all")
    pairable_rows = complete_rows[complete_rows.notna().sum(axis=1) >= 2]
    if pairable_rows.empty:
        exact_agreement = np.nan
        adjacent_agreement = np.nan
    else:
        score_ranges = pairable_rows.max(axis=1) - pairable_rows.min(axis=1)
        exact_agreement = float((score_ranges == 0).mean() * 100.0)
        adjacent_agreement = float((score_ranges <= 1).mean() * 100.0)

    qwk_values = []
    ac2_values = []
    for expert_a, expert_b in itertools.combinations(pivot.columns, 2):
        pair = pivot[[expert_a, expert_b]].dropna()
        if len(pair) < 2:
            continue
        arr_a, arr_b = pair[expert_a].to_numpy(), pair[expert_b].to_numpy()
        qwk_values.append(quadratic_weighted_kappa(arr_a, arr_b))
        ac2_values.append(gwet_ac2_quadratic(arr_a, arr_b))
    finite_qwk = [v for v in qwk_values if np.isfinite(v)]
    finite_ac2 = [v for v in ac2_values if np.isfinite(v)]
    mean_pairwise_qwk = float(np.mean(finite_qwk)) if finite_qwk else np.nan
    mean_pairwise_ac2 = float(np.mean(finite_ac2)) if finite_ac2 else np.nan

    return {
        "agreement_units": int(len(pairable_rows)),
        "exact_agreement": exact_agreement,
        "adjacent_agreement": adjacent_agreement,
        "mean_pairwise_qwk": mean_pairwise_qwk,
        "mean_pairwise_ac2": mean_pairwise_ac2,
        "krippendorff_alpha": krippendorff_alpha_interval(pivot),
    }


def describe_scores(ratings: pd.DataFrame) -> dict[str, float]:
    scores = ratings["score"].astype(float)
    if scores.empty:
        return {
            "n_ratings": 0,
            "n_documents": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "pct_ge4": np.nan,
            "pct_eq5": np.nan,
        }
    return {
        "n_ratings": int(scores.count()),
        "n_documents": int(ratings["doi_norm"].nunique()),
        "mean": float(scores.mean()),
        "sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "median": float(scores.median()),
        "q1": float(scores.quantile(0.25)),
        "q3": float(scores.quantile(0.75)),
        "pct_ge4": float((scores >= 4).mean() * 100.0),
        "pct_eq5": float((scores == 5).mean() * 100.0),
    }


def build_summary_table(ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_row(artifact: str, criterion: str, subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        row = {"artifact_type": artifact, "criterion": criterion}
        row.update(describe_scores(subset))
        row.update(agreement_stats(subset))
        rows.append(row)

    artifacts = sort_artifacts(ratings["artifact_type"].dropna().unique().tolist())
    for artifact in artifacts:
        artifact_subset = ratings[ratings["artifact_type"] == artifact]
        for criterion in CRITERIA:
            add_row(artifact, criterion, artifact_subset[artifact_subset["criterion"] == criterion])
        add_row(artifact, "Overall", artifact_subset)

    if len(artifacts) > 1:
        for criterion in CRITERIA:
            add_row("All", criterion, ratings[ratings["criterion"] == criterion])
        add_row("All", "Overall", ratings)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["artifact_label"] = summary["artifact_type"].map(lambda value: display_artifact(value) if value != "All" else "All evaluation targets")
        column_order = [
            "artifact_type",
            "artifact_label",
            "criterion",
            "n_ratings",
            "n_documents",
            "mean",
            "sd",
            "median",
            "q1",
            "q3",
            "pct_ge4",
            "pct_eq5",
            "agreement_units",
            "exact_agreement",
            "adjacent_agreement",
            "mean_pairwise_qwk",
            "mean_pairwise_ac2",
            "krippendorff_alpha",
        ]
        summary = summary[column_order]
    return summary


def save_figure(fig: plt.Figure, output_base: Path, dpi: int, save_pdf: bool) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    if save_pdf:
        fig.savefig(output_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_sizes(compact: bool) -> dict[str, float]:
    if compact:
        return {
            "title": 12.0,
            "axis": 10.8,
            "tick": 10.2,
            "ytick": 11.0,
            "label": 10.0,
            "small": 9.2,
            "legend": 12.0,
            "panel": 14.5,
        }
    return {
        "title": 12.2,
        "axis": 11.0,
        "tick": 10.5,
        "ytick": 11.5,
        "label": 10.2,
        "small": 8.8,
        "legend": 11.0,
        "panel": 14.8,
    }


def draw_panel_label(axis: plt.Axes, label: str | None, fonts: dict[str, float], x: float = -0.12, y: float = 1.12) -> None:
    if label:
        axis.text(
            x,
            y,
            label,
            transform=axis.transAxes,
            fontsize=fonts["panel"],
            fontweight="bold",
            va="top",
            ha="left",
        )


def draw_likert_distribution_axis(
    axis: plt.Axes,
    artifact_subset: pd.DataFrame,
    artifact: str,
    expert_ids: list[str],
    compact: bool = False,
    show_legend: bool = True,
    panel_label: str | None = None,
) -> None:
    fonts = panel_sizes(compact)
    marker_styles = ["o", "s", "^", "D", "v", "P"]
    y_positions = np.arange(len(CRITERIA))
    bar_height = 0.56 if compact else 0.58
    label_threshold = 8.0 if compact else 7.0
    expert_marker_size = 36 if compact else 30

    for criterion_index, criterion in enumerate(CRITERIA):
        criterion_subset = artifact_subset[artifact_subset["criterion"] == criterion]
        criterion_scores = criterion_subset["score"]
        total_count = int(criterion_scores.count())
        left = 0.0
        for score_value in [1, 2, 3, 4, 5]:
            percentage = float((criterion_scores == score_value).sum() / total_count * 100.0) if total_count else 0.0
            axis.barh(
                criterion_index,
                percentage,
                left=left,
                height=bar_height,
                color=LIKERT_COLORS[score_value],
                edgecolor="white",
                linewidth=0.45,
            )
            if percentage > 0:
                if percentage >= label_threshold:
                    axis.text(
                        left + percentage / 2,
                        criterion_index,
                        f"{percentage:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=fonts["label"],
                        fontweight="medium",
                        color="#1a1a1a" if score_value in {2, 3, 4} else "white",
                    )
                else:
                    axis.text(
                        left + percentage / 2,
                        criterion_index - 0.39,
                        f"{percentage:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=fonts["small"],
                        color="#333333",
                    )
            left += percentage

        pct_high = float((criterion_scores >= 4).mean() * 100.0) if total_count else np.nan
        if np.isfinite(pct_high):
            axis.text(
                102.0,
                criterion_index,
                f"{pct_high:.1f}%",
                va="center",
                fontsize=fonts["axis"],
                fontweight="bold",
                color="#2166ac",
            )

        for expert_index, expert_id in enumerate(expert_ids):
            expert_scores = criterion_subset.loc[criterion_subset["expert_id"] == expert_id, "score"]
            if expert_scores.empty:
                continue
            expert_pct_high = float((expert_scores >= 4).mean() * 100.0)
            expert_y = criterion_index + 0.34 + (expert_index - (len(expert_ids) - 1) / 2) * 0.08
            axis.scatter(
                expert_pct_high,
                expert_y,
                s=expert_marker_size,
                marker=marker_styles[expert_index % len(marker_styles)],
                facecolor="white",
                edgecolor="#222222",
                linewidth=0.8,
                zorder=5,
            )

    n_per_criterion = len(artifact_subset) // max(len(CRITERIA), 1)
    axis.set_title(
        f"{display_artifact(artifact)}\n"
        f"3 experts \u00d7 {artifact_subset['doi_norm'].nunique()} papers = {n_per_criterion} ratings per criterion",
        fontsize=fonts["title"],
        fontweight="bold",
        pad=8,
    )
    axis.set_xlim(0, 116)
    axis.set_ylim(len(CRITERIA) - 0.45, -0.82)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("Share of ratings (%)", fontsize=fonts["axis"], labelpad=5)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(CRITERIA, fontsize=fonts["ytick"])
    axis.tick_params(axis="x", labelsize=fonts["tick"], length=0)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", alpha=0.18, linewidth=0.5, color="#999999")
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.text(
        102.0,
        -0.62,
        "Overall \u22654",
        va="center",
        fontsize=fonts["small"],
        fontweight="bold",
        color="#555555",
        transform=axis.transData,
    )
    draw_panel_label(axis, panel_label, fonts)

    if show_legend:
        score_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=LIKERT_COLORS[score],
                ec="#9a9a9a" if score == 3 else "white",
                linewidth=0.9 if score == 3 else 0.3,
            )
            for score in [1, 2, 3, 4, 5]
        ]
        expert_handles = [
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker=marker_styles[index % len(marker_styles)],
                color="none",
                markerfacecolor="white",
                markeredgecolor="#222222",
                markeredgewidth=0.8,
                markersize=12.0 if compact else 10.5,
                linestyle="None",
            )
            for index, _ in enumerate(expert_ids)
        ]
        score_legend = axis.legend(
            score_handles,
            ["1", "2", "3", "4", "5"],
            title="Likert score",
            title_fontproperties={"weight": "bold", "size": fonts["legend"]},
            loc="upper center",
            bbox_to_anchor=(0.33, -0.27 if compact else -0.20),
            ncol=5,
            frameon=True,
            fancybox=True,
            framealpha=0.75,
            edgecolor="#cccccc",
            fontsize=fonts["legend"],
            handlelength=1.35,
            handleheight=1.0,
            handletextpad=0.62,
            columnspacing=1.05,
        )
        score_legend.get_frame().set_linewidth(0.5)
        axis.add_artist(score_legend)

        expert_legend = axis.legend(
            expert_handles,
            expert_ids,
            title="Expert \u22654 markers",
            title_fontproperties={"weight": "bold", "size": fonts["legend"]},
            loc="upper center",
            bbox_to_anchor=(0.76, -0.27 if compact else -0.20),
            ncol=len(expert_ids),
            frameon=True,
            fancybox=True,
            framealpha=0.75,
            edgecolor="#cccccc",
            fontsize=fonts["legend"],
            handlelength=1.35,
            handleheight=1.0,
            handletextpad=0.65,
            columnspacing=1.10,
        )
        expert_legend.get_frame().set_linewidth(0.5)


def plot_likert_distribution(ratings: pd.DataFrame, output_dir: Path, dpi: int, save_pdf: bool) -> None:
    artifacts = sort_artifacts(ratings["artifact_type"].dropna().unique().tolist())
    expert_ids = sorted(ratings["expert_id"].dropna().unique().tolist())

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}):
        fig_width = max(7.6, 5.4 * len(artifacts))
        fig, axes = plt.subplots(1, len(artifacts), figsize=(fig_width, 4.55), sharey=True)
        if len(artifacts) == 1:
            axes = [axes]

        for axis_index, (axis, artifact) in enumerate(zip(axes, artifacts)):
            artifact_subset = ratings[ratings["artifact_type"] == artifact]
            draw_likert_distribution_axis(
                axis,
                artifact_subset,
                artifact,
                expert_ids,
                compact=False,
                show_legend=axis_index == 0,
            )

        fig.suptitle(
            "Distribution of Domain-Expert Ratings for FRAME Benchmark Datapoints",
            fontsize=13.2,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0.11, 1, 0.91])
        save_figure(fig, output_dir / "figure_1_likert_distribution", dpi, save_pdf)


def reliability_rows(ratings: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifacts = sort_artifacts(ratings["artifact_type"].dropna().unique().tolist())
    expert_ids = sorted(ratings["expert_id"].dropna().unique().tolist())
    expert_pairs = list(itertools.combinations(expert_ids, 2))
    pair_labels = [f"{expert_a} vs {expert_b}" for expert_a, expert_b in expert_pairs]

    row_labels = []
    ac2_matrix = []
    mean_ac2_values = []
    exact_values = []
    adjacent_values = []

    for artifact in artifacts:
        for criterion in CRITERIA:
            subset = ratings[(ratings["artifact_type"] == artifact) & (ratings["criterion"] == criterion)]
            if subset.empty:
                continue
            if len(artifacts) == 1:
                row_labels.append(criterion)
            else:
                row_labels.append(f"{display_artifact_short(artifact)}\n{criterion}")
            pivot = make_pivot(subset)
            pair_ac2 = []
            for expert_a, expert_b in expert_pairs:
                pair = pivot[[expert_a, expert_b]].dropna() if expert_a in pivot and expert_b in pivot else pd.DataFrame()
                if len(pair) < 2:
                    pair_ac2.append(np.nan)
                else:
                    pair_ac2.append(gwet_ac2_quadratic(pair[expert_a].to_numpy(), pair[expert_b].to_numpy()))
            ac2_matrix.append(pair_ac2)
            finite = [v for v in pair_ac2 if np.isfinite(v)]
            mean_ac2_values.append(float(np.mean(finite)) if finite else np.nan)
            stats = agreement_stats(subset)
            exact_values.append(stats["exact_agreement"])
            adjacent_values.append(stats["adjacent_agreement"])

    return (
        row_labels,
        pair_labels,
        np.array(ac2_matrix, dtype=float),
        np.array(mean_ac2_values, dtype=float),
        np.array(exact_values, dtype=float),
        np.array(adjacent_values, dtype=float),
    )


def annotate_heatmap(axis: plt.Axes, matrix: np.ndarray, text_color: str = "black", fontsize: float = 13) -> None:
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=fontsize, color=text_color)


def draw_reliability_panel(
    fig: plt.Figure,
    grid: Any,
    ratings: pd.DataFrame,
    compact: bool = False,
    panel_label: str | None = None,
    agreement_panel_label: str | None = None,
) -> None:
    row_labels, pair_labels, ac2_matrix, mean_ac2_values, exact_values, adjacent_values = reliability_rows(ratings)
    if not row_labels:
        return

    fonts = panel_sizes(compact)
    color_map = plt.get_cmap("RdYlBu")
    ac2_vmin, ac2_vmax = 0.5, 1.0
    ac2_norm = matplotlib.colors.Normalize(vmin=ac2_vmin, vmax=ac2_vmax)

    axis_ac2 = fig.add_subplot(grid[0, 0])
    if pair_labels:
        ac2_panel = np.column_stack([ac2_matrix, mean_ac2_values])
        ac2_labels = pair_labels + ["Mean AC2"]
        n_rows, n_cols = ac2_panel.shape
        # Draw each cell as an explicit solid-color rectangle (PDF-safe)
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = ac2_panel[ri, ci]
                if np.isfinite(val):
                    rgba = color_map(ac2_norm(val))
                else:
                    rgba = "#eeeeee"
                rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1, facecolor=rgba, edgecolor="none")
                axis_ac2.add_patch(rect)
        axis_ac2.set_xlim(-0.5, n_cols - 0.5)
        axis_ac2.set_ylim(n_rows - 0.5, -0.5)
        annotate_heatmap(axis_ac2, ac2_panel, text_color="black", fontsize=fonts["small"])
        axis_ac2.axvline(len(pair_labels) - 0.5, color="white", linewidth=1.4)
        axis_ac2.set_xticks(np.arange(len(ac2_labels)))
        axis_ac2.set_xticklabels(ac2_labels, rotation=30, ha="right", fontsize=fonts["small"])
        axis_ac2.set_yticks(np.arange(len(row_labels)))
        axis_ac2.set_yticklabels(row_labels, fontsize=fonts["tick"])
        axis_ac2.set_title("Gwet's AC2 (pairwise and mean)", fontsize=fonts["title"], fontweight="bold", pad=7)
        sm = matplotlib.cm.ScalarMappable(cmap=color_map, norm=ac2_norm)
        sm.set_array([])
        colorbar = fig.colorbar(sm, ax=axis_ac2, fraction=0.050, pad=0.02)
        colorbar.ax.tick_params(labelsize=fonts["small"])
        colorbar.set_label("AC2", fontsize=fonts["small"], labelpad=4)
    else:
        axis_ac2.text(0.5, 0.5, "At least two experts are required", ha="center", va="center", fontsize=fonts["axis"])
        axis_ac2.axis("off")
    axis_ac2.tick_params(axis="both", length=0)
    for spine in axis_ac2.spines.values():
        spine.set_visible(False)
    draw_panel_label(axis_ac2, panel_label, fonts, x=-0.20 if compact else -0.16, y=1.20 if compact else 1.16)

    axis_agreement = fig.add_subplot(grid[0, 1])
    y_positions = np.arange(len(row_labels))
    bar_height = 0.28 if compact else 0.30
    axis_agreement.barh(y_positions - 0.16, exact_values, height=bar_height, color="#999999", label="Exact")
    axis_agreement.barh(y_positions + 0.16, adjacent_values, height=bar_height, color="#009E73", label="Adjacent")
    for idx, (ex, adj) in enumerate(zip(exact_values, adjacent_values)):
        if np.isfinite(ex):
            axis_agreement.text(ex + 1.5, idx - 0.16, f"{ex:.0f}%", va="center", fontsize=fonts["small"])
        if np.isfinite(adj):
            axis_agreement.text(adj + 1.5, idx + 0.16, f"{adj:.0f}%", va="center", fontsize=fonts["small"])
    axis_agreement.set_xlim(0, 115)
    axis_agreement.set_xticks([0, 25, 50, 75, 100])
    axis_agreement.set_yticks(y_positions)
    axis_agreement.set_yticklabels([])
    axis_agreement.invert_yaxis()
    axis_agreement.set_xlabel("Agreement (%)", fontsize=fonts["axis"])
    axis_agreement.set_title("Exact and adjacent", fontsize=fonts["title"], fontweight="bold", pad=7)
    axis_agreement.grid(axis="x", alpha=0.20, linewidth=0.5, color="#999999")
    axis_agreement.tick_params(axis="x", labelsize=fonts["small"], length=0)
    axis_agreement.tick_params(axis="y", length=0)
    axis_agreement.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=fonts["legend"])
    for spine in axis_agreement.spines.values():
        spine.set_visible(False)
    draw_panel_label(axis_agreement, agreement_panel_label, fonts, x=-0.14 if compact else -0.12, y=1.20 if compact else 1.16)


def plot_reliability(ratings: pd.DataFrame, output_dir: Path, dpi: int, save_pdf: bool) -> None:
    row_labels, pair_labels, _, _, _, _ = reliability_rows(ratings)
    if not row_labels:
        return

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}):
        fig_height = max(4.2, 0.38 * len(row_labels) + 2.7)
        fig = plt.figure(figsize=(8.4, fig_height))
        grid = fig.add_gridspec(1, 2, width_ratios=[max(3.1, len(pair_labels) * 0.95 + 0.7), 2.3], wspace=0.30)
        draw_reliability_panel(fig, grid, ratings, compact=False)
        fig.suptitle("Inter-Expert Agreement for FRAME Human Validation", fontsize=13.2, fontweight="bold", y=0.97)
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.82, wspace=0.28)
        save_figure(fig, output_dir / "figure_2_inter_expert_reliability", dpi, save_pdf)


def shorten_title(title: Any, max_chars: int) -> str:
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return ""
    return textwrap.shorten(str(title).replace("\n", " "), width=max_chars, placeholder="...")


def document_quality_arrays(ratings: pd.DataFrame, max_title_chars: int) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    artifacts = sort_artifacts(ratings["artifact_type"].dropna().unique().tolist())
    columns = [(artifact, criterion) for artifact in artifacts for criterion in CRITERIA]
    if not columns:
        return [], [], np.empty((0, 0)), np.empty((0, 0), dtype=bool)

    mean_scores = (
        ratings.groupby(["doi_norm", "artifact_type", "criterion"], observed=True)["score"]
        .mean()
        .reset_index()
    )
    score_ranges = (
        ratings.groupby(["doi_norm", "artifact_type", "criterion"], observed=True)["score"]
        .agg(lambda values: float(values.max() - values.min()))
        .reset_index(name="score_range")
    )
    titles = ratings.groupby("doi_norm", observed=True)["title"].agg(lambda values: next((value for value in values if isinstance(value, str) and value.strip()), ""))

    documents = sorted(mean_scores["doi_norm"].unique().tolist())
    matrix = np.full((len(documents), len(columns)), np.nan)
    disagreement = np.zeros_like(matrix, dtype=bool)
    column_lookup = {column: column_index for column_index, column in enumerate(columns)}
    document_lookup = {doi_norm: row_index for row_index, doi_norm in enumerate(documents)}

    for _, row in mean_scores.iterrows():
        column_key = (row["artifact_type"], row["criterion"])
        if column_key not in column_lookup:
            continue
        matrix[document_lookup[row["doi_norm"]], column_lookup[column_key]] = row["score"]
    for _, row in score_ranges.iterrows():
        column_key = (row["artifact_type"], row["criterion"])
        if column_key not in column_lookup:
            continue
        disagreement[document_lookup[row["doi_norm"]], column_lookup[column_key]] = row["score_range"] >= 2

    row_means = np.nanmean(matrix, axis=1)
    sort_order = np.argsort(row_means)
    documents = [documents[index] for index in sort_order]
    matrix = matrix[sort_order, :]
    disagreement = disagreement[sort_order, :]

    row_labels = []
    for doi_norm in documents:
        title = shorten_title(titles.get(doi_norm, ""), max_title_chars)
        row_labels.append(title if title else doi_norm)

    if len(artifacts) == 1:
        column_labels = [criterion for _, criterion in columns]
    else:
        column_labels = [f"{display_artifact_short(artifact)}\n{criterion}" for artifact, criterion in columns]
    return row_labels, column_labels, matrix, disagreement


def draw_document_quality_axis(
    fig: plt.Figure,
    axis: plt.Axes,
    ratings: pd.DataFrame,
    max_title_chars: int,
    compact: bool = False,
    panel_label: str | None = None,
) -> None:
    row_labels, column_labels, matrix, disagreement = document_quality_arrays(ratings, max_title_chars)
    if matrix.size == 0:
        axis.axis("off")
        return

    fonts = panel_sizes(compact)
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list("frame_quality", HEATMAP_COLORS, N=256)
    color_map.set_bad("#eeeeee")
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=color_map, vmin=1, vmax=5, aspect="auto")

    axis.set_xticks(np.arange(len(column_labels)))
    axis.set_xticklabels(column_labels, rotation=28, ha="right", fontsize=fonts["tick"] + 1)
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=fonts["small"] + 2)
    axis.set_title("Document-Level Quality Landscape", fontsize=fonts["title"], fontweight="bold", pad=8)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(column_index, row_index, f"{value:.1f}", ha="center", va="center", fontsize=fonts["label"], fontweight="medium")
            if disagreement[row_index, column_index]:
                rectangle = plt.Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.8,
                )
                axis.add_patch(rectangle)

    axis.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(axis="both", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axis, label="Mean expert score", fraction=0.026, pad=0.02)
    colorbar.ax.tick_params(labelsize=fonts["small"] + 2)
    colorbar.set_label("Mean expert score", fontsize=fonts["axis"] + 3, labelpad=6)
    if np.any(disagreement):
        disagreement_handle = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.8)
        axis.legend(
            [disagreement_handle],
            ["Score range \u22652 across experts"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19 if compact else -0.15),
            frameon=False,
            fontsize=fonts["legend"] + 2,
        )
    draw_panel_label(axis, panel_label, fonts, x=-0.08 if compact else -0.06, y=1.10)


def plot_document_quality(ratings: pd.DataFrame, output_dir: Path, dpi: int, save_pdf: bool, max_title_chars: int) -> None:
    row_labels, column_labels, _, _ = document_quality_arrays(ratings, max_title_chars)
    if not row_labels or not column_labels:
        return

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}):
        fig_height = min(28, max(5.8, 0.40 * len(row_labels) + 2.2))
        fig_width = max(9.2, 1.05 * len(column_labels) + 5.3)
        fig, axis = plt.subplots(figsize=(fig_width, fig_height))
        draw_document_quality_axis(fig, axis, ratings, max_title_chars, compact=False)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        save_figure(fig, output_dir / "figure_3_document_quality_landscape", dpi, save_pdf)


def plot_combined_human_eval_panels(ratings: pd.DataFrame, output_dir: Path, dpi: int, save_pdf: bool, max_title_chars: int) -> None:
    artifacts = sort_artifacts(ratings["artifact_type"].dropna().unique().tolist())
    if not artifacts:
        return
    expert_ids = sorted(ratings["expert_id"].dropna().unique().tolist())

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}):
        fig = plt.figure(figsize=(10.2, 8.85))
        outer_grid = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.0, 0.94],
            width_ratios=[1.16, 1.0],
            hspace=0.92,
            wspace=0.28,
        )

        axis_likert = fig.add_subplot(outer_grid[0, :])
        artifact = artifacts[0]
        artifact_subset = ratings[ratings["artifact_type"] == artifact]
        draw_likert_distribution_axis(
            axis_likert,
            artifact_subset,
            artifact,
            expert_ids,
            compact=True,
            show_legend=True,
            panel_label="A",
        )

        reliability_grid = outer_grid[1, :].subgridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.28)
        draw_reliability_panel(fig, reliability_grid, ratings, compact=True, panel_label="B", agreement_panel_label="C")

        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.97)
        save_figure(fig, output_dir / "figure_4_human_eval_combined_panels", dpi, save_pdf)


def format_number(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def write_markdown_table(summary: pd.DataFrame, output_path: Path) -> None:
    display_columns = [
        "artifact_label",
        "criterion",
        "mean",
        "sd",
        "median",
        "q1",
        "q3",
        "pct_ge4",
        "exact_agreement",
        "adjacent_agreement",
        "mean_pairwise_ac2",
    ]
    headers = [
        "Evaluation target",
        "Criterion",
        "Mean (1-5)",
        "SD",
        "Median",
        "Q1",
        "Q3",
        "% >=4",
        "Exact %",
        "Adjacent %",
        "Mean AC2",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in summary.iterrows():
        cells = []
        for column in display_columns:
            value = row[column]
            if column in {"artifact_label", "criterion"}:
                cells.append(str(value))
            elif column in {"pct_ge4", "exact_agreement", "adjacent_agreement"}:
                cells.append(format_number(value, 1))
            else:
                cells.append(format_number(value, 2))
        lines.append("| " + " | ".join(cells) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_report(ratings: pd.DataFrame, benchmark_keys: set[str], output_path: Path) -> None:
    unmatched = sorted(set(ratings["doi_norm"]) - benchmark_keys) if benchmark_keys else []
    duplicate_mask = ratings.duplicated(UNIT_COLUMNS + ["expert_id"], keep=False)
    duplicate_count = int(duplicate_mask.sum())

    expected_index = pd.MultiIndex.from_product(
        [
            sorted(ratings["doi_norm"].unique()),
            sort_artifacts(ratings["artifact_type"].unique().tolist()),
            CRITERIA,
            sorted(ratings["expert_id"].unique()),
        ],
        names=UNIT_COLUMNS + ["expert_id"],
    )
    observed_index = pd.MultiIndex.from_frame(ratings[UNIT_COLUMNS + ["expert_id"]].drop_duplicates())
    missing_index = expected_index.difference(observed_index)

    lines = [
        "# FRAME Human-Evaluation Validation Report",
        "",
        f"- Ratings loaded: {len(ratings)}",
        f"- Unique documents: {ratings['doi_norm'].nunique()}",
        f"- Experts: {', '.join(sorted(ratings['expert_id'].unique()))}",
        f"- Evaluation targets: {', '.join(display_artifact(artifact) for artifact in sort_artifacts(ratings['artifact_type'].unique().tolist()))}",
        f"- Duplicate expert/unit rows: {duplicate_count}",
        f"- Missing expected expert/unit rows: {len(missing_index)}",
    ]
    if benchmark_keys:
        lines.append(f"- DOI keys matched to benchmark JSON: {ratings['doi_norm'].nunique() - len(unmatched)} / {ratings['doi_norm'].nunique()}")
    if unmatched:
        lines.extend(["", "## Unmatched DOI Keys", ""])
        lines.extend(f"- {doi_norm}" for doi_norm in unmatched[:100])
        if len(unmatched) > 100:
            lines.append(f"- ... {len(unmatched) - 100} more")
    if len(missing_index) > 0:
        lines.extend(["", "## First Missing Expert/Unit Rows", ""])
        for missing_key in missing_index[:50]:
            lines.append("- " + " | ".join(str(part) for part in missing_key))
        if len(missing_index) > 50:
            lines.append(f"- ... {len(missing_index) - 50} more")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manuscript_notes(summary: pd.DataFrame, output_path: Path) -> None:
    overall_rows = summary[summary["criterion"] == "Overall"]
    all_target_overall = overall_rows[overall_rows["artifact_type"] == "All"]
    if not all_target_overall.empty:
        stats = all_target_overall.iloc[0].to_dict()
    elif not overall_rows.empty:
        stats = overall_rows.iloc[0].to_dict()
    else:
        stats = {}

    pct_ge4 = format_number(stats.get("pct_ge4", np.nan), 1)
    adjacent = format_number(stats.get("adjacent_agreement", np.nan), 1)
    mean_ac2 = format_number(stats.get("mean_pairwise_ac2", np.nan), 2)

    notes = f"""# Manuscript Notes for FRAME Human Validation

## Suggested Results Text

Three domain experts independently evaluated each FRAME benchmark datapoint as a whole, comparing both `input.txt` and `output.txt` against the source paper PDF. The ratings therefore represent overall input/output extraction quality for each datapoint, rather than separate scores for input keys and output keys. Across all expert ratings, {pct_ge4}% of scores were 4 or 5 on the five-point Likert scale, indicating that the extracted fields were generally judged suitable for benchmark evaluation and VLM fine-tuning. Inter-expert reliability was assessed using Gwet's AC2 with quadratic weights (Gwet, 2014), which is robust to the high-score concentration that makes traditional Kappa metrics unreliable in this setting. The mean pairwise AC2 was {mean_ac2}, and {adjacent}% of comparable rating units differed by no more than one point (adjacent agreement). The full machine-readable summary retains QWK and Krippendorff alpha as supplementary diagnostics, but AC2 is emphasized because kappa-style chance models are unstable for these highly skewed ordinal ratings.

## Figure 1 Caption

Distribution of domain-expert Likert ratings for FRAME benchmark datapoint quality. Each rating evaluates the input and output extraction together for a paper-level benchmark datapoint, grouped by Correctness, Completeness, Coherence, and Faithfulness. The annotation at the end of each row reports the percentage of ratings greater than or equal to 4, while open markers indicate the corresponding high-score share for each individual expert.

## Figure 2 Caption

Inter-expert agreement for FRAME human validation. The AC2 heatmap reports pairwise Gwet's AC2 with quadratic weights for each expert pair, with an additional mean AC2 column for each criterion. The agreement panel reports exact agreement and adjacent agreement. Adjacent agreement counts rating units for which expert scores differ by no more than one point on the 1-5 scale.

## Figure 3 Caption

Document-level quality landscape for the FRAME benchmark subset evaluated by domain experts. Rows correspond to evaluated paper titles, columns correspond to overall input/output extraction criteria, and each cell reports the mean expert rating for the paper-level datapoint. Cells outlined in black indicate score ranges of at least two points across experts, highlighting papers or criteria that may need manual review.

## Combined Figure Caption

Human validation of FRAME benchmark datapoints. (A) Distribution of expert Likert ratings by criterion, with end-of-row annotations showing the percentage of ratings greater than or equal to 4 and open markers showing individual expert high-score shares. (B) Pairwise and mean Gwet's AC2 by criterion. (C) Exact and adjacent inter-expert agreement by criterion.

## Table Caption

Human validation summary for FRAME benchmark datapoints. Each row summarizes expert ratings for a qualitative criterion. Ratings assess agreement between the combined input/output benchmark extraction and the original scientific paper, not the performance of a downstream VLM on the benchmark. Gwet's AC2 is reported as the primary chance-corrected agreement metric; QWK and Krippendorff alpha are retained in the CSV summary for reference.
"""
    output_path.write_text(notes, encoding="utf-8")


def write_template(output_path: Path) -> None:
    template = pd.DataFrame(
        [
            {
                "expert_id": "expert_1",
                "doi_norm": "10_1007_example_2026_001",
                "title": "Example paper title",
                "artifact_type": "combined",
                "Correctness": "",
                "Completeness": "",
                "Coherence": "",
                "Faithfulness": "",
                "notes": "optional",
            }
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)


def run_analysis(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_keys = load_benchmark_keys(Path(args.final_input), Path(args.final_output))
    ratings = load_ratings([Path(path) for path in args.ratings], args.default_artifact_type, args.expert_id)
    ratings.to_csv(output_dir / "normalized_human_ratings.csv", index=False)

    write_validation_report(ratings, benchmark_keys, output_dir / "human_eval_validation_report.md")
    summary = build_summary_table(ratings)
    summary.to_csv(output_dir / "human_eval_summary_table.csv", index=False)
    write_markdown_table(summary, output_dir / "human_eval_summary_table.md")
    write_manuscript_notes(summary, output_dir / "human_eval_manuscript_notes.md")

    plot_likert_distribution(ratings, output_dir, args.dpi, not args.no_pdf)
    plot_reliability(ratings, output_dir, args.dpi, not args.no_pdf)
    plot_document_quality(ratings, output_dir, args.dpi, not args.no_pdf, args.max_title_chars)
    plot_combined_human_eval_panels(ratings, output_dir, args.dpi, not args.no_pdf, args.max_title_chars)

    print("FRAME human-evaluation analysis complete")
    print(f"Ratings: {len(ratings)}")
    print(f"Documents: {ratings['doi_norm'].nunique()}")
    print(f"Experts: {', '.join(sorted(ratings['expert_id'].unique()))}")
    print(f"Outputs: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FRAME human-evaluation figures and summary tables.")
    parser.add_argument("--ratings", nargs="+", help="CSV/TSV/XLSX expert rating files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for figures, tables, and normalized ratings.")
    parser.add_argument("--final-input", default=str(DEFAULT_FINAL_INPUT), help="Path to final_input.json for DOI validation.")
    parser.add_argument("--final-output", default=str(DEFAULT_FINAL_OUTPUT), help="Path to final_output.json for DOI validation.")
    parser.add_argument(
        "--default-artifact-type",
        choices=["input", "output", "combined"],
        default=None,
        help="Evaluation target to use when it cannot be inferred; use combined for one overall input/output datapoint score.",
    )
    parser.add_argument("--expert-id", default=None, help="Expert ID to assign when processing a single rating file without an expert column.")
    parser.add_argument("--write-template", default=None, help="Write a CSV template to this path and exit.")
    parser.add_argument("--dpi", type=int, default=450, help="Figure DPI for PNG/PDF export.")
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures, not PDF figures.")
    parser.add_argument("--max-title-chars", type=int, default=72, help="Maximum title length shown in the document heatmap.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.write_template:
        write_template(Path(args.write_template))
        print(f"Wrote template: {args.write_template}")
        return
    if not args.ratings:
        parser.error("--ratings is required unless --write-template is used")
    run_analysis(args)


if __name__ == "__main__":
    main()