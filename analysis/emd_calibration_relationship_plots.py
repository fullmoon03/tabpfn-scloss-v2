from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_correlation_summary(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for metric in ["nll", "ece"]:
        pearson = pearsonr(details["emd"].to_numpy(), details[metric].to_numpy())
        spearman = spearmanr(details["emd"].to_numpy(), details[metric].to_numpy())
        rows.append(
            {
                "x_metric": "emd",
                "y_metric": metric,
                "pearson_r": float(pearson.statistic),
                "pearson_pvalue": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
            }
        )
    return pd.DataFrame(rows)


def plot_scatter(
    details: pd.DataFrame,
    y_metric: str,
    output_path: Path,
) -> None:
    x = details["emd"].to_numpy(dtype=np.float64)
    y = details[y_metric].to_numpy(dtype=np.float64)
    pearson = pearsonr(x, y).statistic
    spearman = spearmanr(x, y).statistic

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(x, y, s=38, alpha=0.82, color="#1f5aa6")
    ax.set_xlabel("EMD")
    ax.set_ylabel(y_metric.upper() if y_metric == "ece" else "NLL")
    ax.set_title(f"EMD vs {y_metric.upper() if y_metric == 'ece' else 'NLL'}")
    ax.text(
        0.03,
        0.97,
        f"Pearson r={pearson:.3f}\nSpearman rho={spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_note(correlations: pd.DataFrame, details: pd.DataFrame, output_path: Path) -> None:
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
            f"- EMD vs {row['y_metric'].upper() if row['y_metric']=='ece' else 'NLL'}: "
            f"Pearson r={row['pearson_r']:.3f}, Spearman rho={row['spearman_rho']:.3f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_from_details_csv(details_csv: str | Path) -> None:
    details_csv = Path(details_csv)
    outdir = details_csv.parent
    details = pd.read_csv(details_csv)
    details = details.rename(
        columns={
            "EMD": "emd",
            "NLL": "nll",
            "ECE": "ece",
            "Accuracy": "accuracy",
        }
    )
    correlations = compute_correlation_summary(details)
    correlations.to_csv(outdir / "correlations.csv", index=False)
    plot_scatter(details, "nll", outdir / "scatter_emd_nll.png")
    plot_scatter(details, "ece", outdir / "scatter_emd_ece.png")
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
