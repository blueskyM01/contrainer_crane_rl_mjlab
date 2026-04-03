"""Generate per-group and global summary plots from batch_eval_summary.csv.

For each unique run directory found in the summary CSV, writes:
  <run_dir>/group_eval_summary.csv  — filtered rows for that run group
  <run_dir>/group_eval_summary.png  — metrics vs iteration plot

Also writes a global plot alongside the summary CSV:
  <summary_csv_dir>/batch_eval_summary.png

If sway metrics are not present in the summary CSV, this script backfills them from
each results directory's track_pos_eval.txt when available.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

import mjlab

METRICS: list[str] = [
    "mae",
    "rmse",
    "max_abs_error",
    "nrmse_range",
    "sway_abs_mae_deg",
    "sway_rms_deg",
    "sway_max_abs_deg",
]
METRIC_LABELS: dict[str, str] = {
    "mae": "MAE",
    "rmse": "RMSE",
    "max_abs_error": "MaxAbsError",
    "nrmse_range": "NRMSE(range)",
    "sway_abs_mae_deg": "SwayAbsMAE (deg)",
    "sway_rms_deg": "SwayRMS (deg)",
    "sway_max_abs_deg": "SwayMaxAbs (deg)",
}
COLORS: dict[str, str] = {
    "mae": "C0",
    "rmse": "C1",
    "max_abs_error": "C2",
    "nrmse_range": "C3",
    "sway_abs_mae_deg": "C4",
    "sway_rms_deg": "C5",
    "sway_max_abs_deg": "C6",
}
TRACK_EVAL_KEYS: dict[str, str] = {
    "SwayAbsMAE(deg)": "sway_abs_mae_deg",
    "SwayRMS(deg)": "sway_rms_deg",
    "SwayMaxAbs(deg)": "sway_max_abs_deg",
}


@dataclass(frozen=True)
class BatchEvalPlotConfig:
    summary_csv: str
    """Path to batch_eval_summary.csv produced by Mjlab-QcAntiSwayAlignment.sh."""


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, AttributeError):
        return None


def _extract_model_id(checkpoint_file: str) -> int:
    """Extract numeric iteration from model_<N>.pt filename."""
    stem = Path(checkpoint_file).stem
    for part in reversed(stem.split("_")):
        if part.isdigit():
            return int(part)
    return -1


def _parse_track_eval_metrics(track_eval_path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not track_eval_path.is_file():
        return metrics

    pattern = re.compile(r"^\[INFO\]\s+([^:]+):\s+(.+?)\s*$")
    for line in track_eval_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        label, value = match.groups()
        key = TRACK_EVAL_KEYS.get(label)
        if key is not None:
            metrics[key] = value
    return metrics


def _augment_rows_with_track_eval(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    augmented_rows: list[dict[str, str]] = []
    for row in rows:
        augmented = dict(row)
        needs_backfill = any(not augmented.get(metric, "").strip() for metric in TRACK_EVAL_KEYS.values())
        if needs_backfill:
            results_dir = augmented.get("results_dir", "")
            if results_dir:
                track_eval_metrics = _parse_track_eval_metrics(Path(results_dir) / "track_pos_eval.txt")
                for key, value in track_eval_metrics.items():
                    if not augmented.get(key, "").strip():
                        augmented[key] = value
        augmented_rows.append(augmented)
    return augmented_rows


def _active_metrics(rows: list[dict[str, str]]) -> list[str]:
    active: list[str] = []
    for metric in METRICS:
        has_value = any(_to_float(row.get(metric, "")) is not None for row in rows)
        if has_value:
            active.append(metric)
    return active


def _sorted_valid_rows(
    rows: list[dict[str, str]],
) -> tuple[list[int], dict[str, list[float]]]:
    """Return (sorted iterations, {metric: [values]}) for rows with eval_status==0."""
    entries: list[tuple[int, dict[str, float]]] = []
    for row in rows:
        if row.get("eval_status", "1").strip() != "0":
            continue
        model_id = _extract_model_id(row.get("checkpoint_file", ""))
        vals: dict[str, float] = {}
        for m in METRICS:
            v = _to_float(row.get(m, ""))
            if v is not None:
                vals[m] = v
        if vals:
            entries.append((model_id, vals))
    entries.sort(key=lambda e: e[0])
    iterations = [e[0] for e in entries]
    metric_vals: dict[str, list[float]] = {m: [] for m in METRICS}
    for _, vals in entries:
        for m in METRICS:
            metric_vals[m].append(vals.get(m, float("nan")))
    return iterations, metric_vals


def _plot_group(
    rows: list[dict[str, str]], title: str, output_png: Path, metrics: list[str]
) -> None:
    iterations, metric_vals = _sorted_valid_rows(rows)
    if not iterations:
        print(f"[WARN] No valid eval rows to plot for: {title}")
        return

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        ys = metric_vals[m]
        ax.plot(iterations, ys, marker="o", color=COLORS[m], label=METRIC_LABELS[m])
        finite = [(x, y) for x, y in zip(iterations, ys) if np.isfinite(y)]
        if finite:
            xs_f, ys_f = zip(*finite)
            min_idx = int(np.argmin(ys_f))
            ax.scatter(
                [xs_f[min_idx]],
                [ys_f[min_idx]],
                color="red",
                zorder=5,
                label=f"Best iter={xs_f[min_idx]} ({ys_f[min_idx]:.4f})",
            )
        ax.set_ylabel(METRIC_LABELS[m])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(str(output_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved group plot: {output_png}")


def _plot_global(
    groups: dict[str, list[dict[str, str]]],
    output_png: Path,
    metrics: list[str],
) -> None:
    """One figure: one subplot per metric, one line per run group."""
    group_names = list(groups.keys())
    cmap = plt.cm.tab10  # type: ignore[attr-defined]
    group_colors = [cmap(i / max(len(group_names), 1)) for i in range(len(group_names))]

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=False
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        for (run_dir, group_rows), color in zip(groups.items(), group_colors):
            iterations, metric_vals = _sorted_valid_rows(group_rows)
            if not iterations:
                continue
            ys = metric_vals[m]
            label = Path(run_dir).name
            ax.plot(iterations, ys, marker="o", color=color, label=label)
        ax.set_ylabel(METRIC_LABELS[m])
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Iteration")

    fig.suptitle("Batch Eval Summary", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(output_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved global plot: {output_png}")


def generate_plots(cfg: BatchEvalPlotConfig) -> None:
    summary_csv = Path(cfg.summary_csv)
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    rows = _read_csv(summary_csv)
    if not rows:
        print("[WARN] No rows in summary CSV, skipping plot generation.")
        return

    rows = _augment_rows_with_track_eval(rows)
    metrics = _active_metrics(rows)
    if not metrics:
        print("[WARN] No numeric metrics found in summary CSV, skipping plot generation.")
        return

    # Group rows by run directory (parent of checkpoint_file).
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        ckpt = row.get("checkpoint_file", "")
        run_dir = str(Path(ckpt).parent) if ckpt else "unknown"
        groups[run_dir].append(row)

    fieldnames = list(rows[0].keys())
    for metric in metrics:
        if metric not in fieldnames:
            fieldnames.append(metric)

    # Per-group CSV and PNG.
    for run_dir, group_rows in groups.items():
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        group_csv = run_path / "group_eval_summary.csv"
        with open(group_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_rows)
        print(f"[INFO] Saved group CSV: {group_csv}")

        group_png = run_path / "group_eval_summary.png"
        _plot_group(group_rows, title=run_path.name, output_png=group_png, metrics=metrics)

    # Global PNG next to the summary CSV.
    global_png = summary_csv.with_suffix(".png")
    _plot_global(groups, output_png=global_png, metrics=metrics)


def main() -> None:
    cfg = tyro.cli(BatchEvalPlotConfig, config=mjlab.TYRO_FLAGS)
    generate_plots(cfg)


if __name__ == "__main__":
    main()
