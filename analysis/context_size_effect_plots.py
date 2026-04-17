from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY_NAMES = {
    "emd": "EMD",
    "nll": "NLL",
    "ece": "ECE",
    "accuracy": "Accuracy",
}


def _normalize_columns(details: pd.DataFrame) -> pd.DataFrame:
    return details.rename(
        columns={
            "EMD": "emd",
            "NLL": "nll",
            "ECE": "ece",
            "Accuracy": "accuracy",
        }
    )


def summarize_by_context_size(details: pd.DataFrame) -> pd.DataFrame:
    grouped = details.groupby("context_size")[["emd", "nll", "ece", "accuracy"]]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    count_df = grouped.size().rename("n_repeats")
    summary = pd.concat([count_df, mean_df, std_df], axis=1).reset_index()
    return summary


def plot_metric_panels(details: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    summary = summarize_by_context_size(details)
    context_sizes = summary["context_size"].to_numpy(dtype=np.int32)

    for ax, metric in zip(axes, ["emd", "accuracy", "nll", "ece"]):
        for context_size, group in details.groupby("context_size"):
            x = np.full(group.shape[0], context_size, dtype=np.float64)
            jitter = np.linspace(-0.9, 0.9, group.shape[0]) if group.shape[0] > 1 else np.array([0.0])
            ax.scatter(
                x + jitter,
                group[metric].to_numpy(dtype=np.float64),
                s=22,
                alpha=0.45,
                color="#94a3b8",
            )

        ax.plot(
            context_sizes,
            summary[f"{metric}_mean"].to_numpy(dtype=np.float64),
            color="#1f5aa6",
            linewidth=2.0,
            marker="o",
            markersize=5,
        )
        std = summary[f"{metric}_std"].to_numpy(dtype=np.float64)
        mean = summary[f"{metric}_mean"].to_numpy(dtype=np.float64)
        ax.fill_between(
            context_sizes,
            mean - std,
            mean + std,
            color="#1f5aa6",
            alpha=0.15,
        )
        ax.set_title(DISPLAY_NAMES[metric])
        ax.set_xlabel("Context size")
        ax.set_ylabel(DISPLAY_NAMES[metric])
        ax.grid(alpha=0.25)

    fig.suptitle("Effect of Context Size on EMD / Accuracy / NLL / ECE", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_note(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Context Size Effect",
        "",
        "Per-context-size summary of EMD, Accuracy, NLL, and ECE.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- context_size={int(row['context_size'])}: "
            f"EMD={row['emd_mean']:.6f}±{row['emd_std']:.6f}, "
            f"Accuracy={row['accuracy_mean']:.6f}±{row['accuracy_std']:.6f}, "
            f"NLL={row['nll_mean']:.6f}±{row['nll_std']:.6f}, "
            f"ECE={row['ece_mean']:.6f}±{row['ece_std']:.6f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_from_details_csv(details_csv: str | Path) -> None:
    details_csv = Path(details_csv)
    outdir = details_csv.parent
    details = _normalize_columns(pd.read_csv(details_csv))

    summary = summarize_by_context_size(details)
    summary.to_csv(outdir / "summary_by_context_size.csv", index=False)
    plot_metric_panels(details, outdir / "metrics_vs_context_size.png")
    write_summary_note(summary, outdir / "summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze effect of context size on EMD / Accuracy / NLL / ECE."
    )
    parser.add_argument("details_csv", type=Path)
    args = parser.parse_args()
    run_from_details_csv(args.details_csv)


if __name__ == "__main__":
    main()
