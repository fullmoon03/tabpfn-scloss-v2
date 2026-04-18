from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY_NAME = {
    "emd": "EMD",
    "nll": "NLL",
    "ece": "ECE",
    "accuracy": "Accuracy",
}
GROUP_COLORS = {
    1: "#1f77b4",
    2: "#2ca02c",
    3: "#ffbf00",
    4: "#d62728",
    5: "#9467bd",
}


def normalize_details_columns(details: pd.DataFrame) -> pd.DataFrame:
    return details.rename(
        columns={
            "EMD": "emd",
            "NLL": "nll",
            "ECE": "ece",
            "Accuracy": "accuracy",
        }
    )


def assign_emd_groups(details: pd.DataFrame, n_groups: int = 5) -> pd.DataFrame:
    details = details.copy()
    order = np.argsort(details["emd"].to_numpy(dtype=np.float64))
    emd_group = np.empty(len(details), dtype=np.int32)
    base_group_size = len(details) // n_groups

    for group_idx in range(n_groups):
        start = group_idx * base_group_size
        end = (group_idx + 1) * base_group_size if group_idx < n_groups - 1 else len(details)
        emd_group[order[start:end]] = group_idx + 1

    details["emd_group"] = emd_group
    return details


def compute_correlation_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for x_metric, y_metric in [("emd", "nll"), ("emd", "ece"), ("nll", "ece")]:
        pearson = pearsonr(details[x_metric].to_numpy(), details[y_metric].to_numpy())
        spearman = spearmanr(
            details[x_metric].to_numpy(),
            details[y_metric].to_numpy(),
        )
        rows.append(
            {
                "x_metric": x_metric,
                "y_metric": y_metric,
                "pearson_r": float(pearson.statistic),
                "pearson_pvalue": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_scatter_triptych(
    details: pd.DataFrame,
    correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    pairs = [
        ("emd", "nll", "EMD vs NLL"),
        ("emd", "ece", "EMD vs ECE"),
        ("nll", "ece", "NLL vs ECE"),
    ]
    corr_lookup = {
        (row["x_metric"], row["y_metric"]): (row["pearson_r"], row["spearman_rho"])
        for _, row in correlations.iterrows()
    }

    for ax, (x_metric, y_metric, title) in zip(axes, pairs):
        for group_idx in range(1, 6):
            sub = details[details["emd_group"] == group_idx]
            ax.scatter(
                sub[x_metric].to_numpy(dtype=np.float64),
                sub[y_metric].to_numpy(dtype=np.float64),
                s=16,
                alpha=0.85,
                color=GROUP_COLORS[group_idx],
                edgecolors="none",
                label=f"EMD G{group_idx}",
            )

        pearson, spearman = corr_lookup[(x_metric, y_metric)]
        ax.text(
            0.03,
            0.97,
            f"Pearson={pearson:.3f}\nSpearman={spearman:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "0.7",
                "boxstyle": "round,pad=0.2",
            },
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(DISPLAY_NAME[x_metric])
        ax.set_ylabel(DISPLAY_NAME[y_metric])
        ax.grid(alpha=0.25)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_COLORS[group_idx],
            markersize=5,
            label=f"EMD G{group_idx}",
        )
        for group_idx in range(1, 6)
    ]
    fig.legend(
        handles=handles,
        labels=[f"EMD G{group_idx}" for group_idx in range(1, 6)],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8,
    )
    fig.suptitle("OpenML 54 relation (repeats)\nEMD-sorted 5 groups", y=1.06, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_note(
    correlations: pd.DataFrame, details: pd.DataFrame, output_path: Path
) -> None:
    lines = [
        "# EMD vs Calibration",
        "",
        f"Repeats: {len(details)}",
        f"Mean EMD: {details['emd'].mean():.6f}",
        f"Mean NLL: {details['nll'].mean():.6f}",
        f"Mean ECE: {details['ece'].mean():.6f}",
        f"Mean Accuracy: {details['accuracy'].mean():.6f}",
        "",
        "## Correlations",
    ]
    for _, row in correlations.iterrows():
        lines.append(
            f"- {DISPLAY_NAME[row['x_metric']]} vs {DISPLAY_NAME[row['y_metric']]}: "
            f"Pearson r={row['pearson_r']:.3f}, Spearman rho={row['spearman_rho']:.3f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_from_details_csv(details_csv: str | Path) -> None:
    details_csv = Path(details_csv)
    outdir = details_csv.parent
    details = normalize_details_columns(pd.read_csv(details_csv))
    details = assign_emd_groups(details)

    correlations = compute_correlation_summary(details)
    correlations.to_csv(outdir / "correlations.csv", index=False)
    plot_grouped_scatter_triptych(
        details,
        correlations,
        outdir / "scatter_emd_nll_ece_grouped.png",
    )
    write_summary_note(correlations, details, outdir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze relationship between EMD and calibration metrics."
    )
    parser.add_argument("details_csv", type=Path)
    args = parser.parse_args()
    run_from_details_csv(args.details_csv)


if __name__ == "__main__":
    main()
