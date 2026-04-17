from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_variance_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    var_columns = [name for name in reader.fieldnames if name.startswith("var_t")]
    if not var_columns:
        raise ValueError(f"No var_t columns found in {csv_path}.")

    var_columns = sorted(var_columns, key=lambda name: int(name.replace("var_t", "")))
    query_ids = np.asarray([int(row["query_id"]) for row in rows], dtype=np.int32)
    y_true = np.asarray([row["y_true"] for row in rows], dtype=object)
    values = np.asarray(
        [[float(row[col]) for col in var_columns] for row in rows], dtype=np.float64
    )
    return query_ids, y_true, values


def plot_query_trajectories(
    query_ids: np.ndarray,
    values: np.ndarray,
    output_path: str | Path,
) -> None:
    t = np.arange(values.shape[1])
    fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for idx, query_id in enumerate(query_ids):
        ax = axes[idx]
        ax.plot(
            t,
            values[idx],
            linewidth=1.6,
            color="#1f5aa6",
        )
        ax.set_title(f"q={query_id}", fontsize=10)
        ax.grid(alpha=0.25)

    for ax in axes[len(query_ids) :]:
        ax.axis("off")

    fig.suptitle("Conditional Theta Variance by Query", y=0.995)
    fig.supxlabel("Step t")
    fig.supylabel("Var_t")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_std(
    values: np.ndarray,
    output_path: str | Path,
) -> None:
    t = np.arange(values.shape[1])
    mean = values.mean(axis=0)
    std = values.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mean, color="#1f5aa6", linewidth=2.2, label="Query mean")
    ax.fill_between(
        t,
        np.clip(mean - std, 0.0, None),
        mean + std,
        color="#1f5aa6",
        alpha=0.18,
        label="±1 std",
    )
    ax.set_title("Mean Conditional Theta Variance Across Queries")
    ax.set_xlabel("Step t")
    ax.set_ylabel("Var_t")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loglog_mean(
    query_ids: np.ndarray,
    values: np.ndarray,
    output_path: str | Path,
) -> None:
    t = np.arange(1, values.shape[1], dtype=np.float64)
    fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for idx, query_id in enumerate(query_ids):
        ax = axes[idx]
        series = values[idx, 1:]
        positive_mask = series > 0
        t_plot = t[positive_mask]
        series_plot = series[positive_mask]
        if t_plot.size == 0:
            ax.set_title(f"q={query_id}", fontsize=10)
            ax.text(0.5, 0.5, "no positive values", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.grid(alpha=0.25, which="both")
            continue

        reference = series_plot[0] * (t_plot[0] / t_plot)
        ax.loglog(t_plot, series_plot, color="#0f766e", linewidth=1.8)
        ax.loglog(
            t_plot,
            reference,
            color="#b45309",
            linewidth=1.2,
            linestyle="--",
        )
        ax.set_title(f"q={query_id}", fontsize=10)
        ax.grid(alpha=0.25, which="both")

    for ax in axes[len(query_ids) :]:
        ax.axis("off")

    fig.suptitle("Log-Log Conditional Theta Variance by Query", y=0.995)
    fig.supxlabel("Step t")
    fig.supylabel("Var_t")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_variance(
    values: np.ndarray,
    output_path: str | Path,
) -> None:
    t = np.arange(values.shape[1])
    per_query_max = values.max(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        per_query_max,
        out=np.zeros_like(values),
        where=per_query_max > 0,
    )
    mean = normalized.mean(axis=0)
    std = normalized.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in normalized:
        ax.plot(t, row, color="#94a3b8", alpha=0.35, linewidth=1.0)

    ax.plot(t, mean, color="#7c3aed", linewidth=2.2, label="Normalized mean")
    ax.fill_between(
        t,
        np.clip(mean - std, 0.0, 1.0),
        np.clip(mean + std, 0.0, 1.0),
        color="#7c3aed",
        alpha=0.18,
        label="±1 std",
    )
    ax.set_title("Relative Conditional Theta Variance")
    ax.set_xlabel("Step t")
    ax.set_ylabel("Normalized Var_t")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create analysis plots from conditional-theta-variance.csv"
    )
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    query_ids, _, values = load_variance_csv(args.csv_path)
    outdir = args.csv_path.parent

    plot_query_trajectories(
        query_ids,
        values,
        outdir / "conditional-theta-variance-query-trajectories.png",
    )
    plot_mean_std(
        values,
        outdir / "conditional-theta-variance-mean-std.png",
    )
    plot_loglog_mean(
        query_ids,
        values,
        outdir / "conditional-theta-variance-loglog.png",
    )
    plot_normalized_variance(
        values,
        outdir / "conditional-theta-variance-relative.png",
    )


if __name__ == "__main__":
    main()
