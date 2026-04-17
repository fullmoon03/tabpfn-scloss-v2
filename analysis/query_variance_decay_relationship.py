from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax

from dgp import OPENML_BINARY_CLASSIFICATION, OPENML_CLASSIFICATION, OPENML_REGRESSION, load_dgp
from fixed_query_experiments.rollout import make_classifier_pred_rule
import utils

EPS = 1e-12
FEATURE_COLUMNS = [
    "init_entropy",
    "local_class_entropy_k5",
    "knn_avg_dist_k5",
    "init_margin_top1_top2",
    "init_true_class_prob",
    "same_diff_gap_k5",
]
SCATTER_FEATURE_COLUMNS = [
    "init_entropy",
    "local_class_entropy_k5",
    "knn_avg_dist_k5",
    "init_margin_top1_top2",
    "init_true_class_prob",
]
HIGHLIGHT_QUERY_IDS = [60, 273, 395, 514, 580, 586, 732]
HIGHLIGHT_COLORS = {
    60: "#d62728",
    273: "#ff7f0e",
    395: "#2ca02c",
    514: "#9467bd",
    580: "#8c564b",
    586: "#e377c2",
    732: "#17becf",
}


def parse_output_metadata(output_dir: Path) -> dict[str, int | str]:
    name = output_dir.name
    patterns = {
        "dgp_name": r"name=(.+?)_data=",
        "data_size": r"_data=(\d+)",
        "num_queries": r"_queries=(\d+)",
        "rollout_length": r"_rollout_length=(\d+)",
        "num_posterior_samples": r"_posterior_samples=(\d+)",
        "seed": r"_seed=(\d+)",
    }
    parsed: dict[str, int | str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if not match:
            raise ValueError(f"Could not parse {key} from output directory name: {name}")
        value = match.group(1)
        parsed[key] = value if key == "dgp_name" else int(value)
    return parsed


def load_analysis_cfg(metadata: dict[str, int | str]):
    cfg = OmegaConf.load("conf/conditional-theta-variance.yaml")
    dgp_name = metadata["dgp_name"]
    dgp_cfg_path = Path("conf/dgp") / f"{dgp_name}.yaml"
    if dgp_cfg_path.exists():
        dgp_cfg = OmegaConf.load(dgp_cfg_path)
    elif dgp_name in OPENML_CLASSIFICATION + OPENML_BINARY_CLASSIFICATION + OPENML_REGRESSION:
        dgp_cfg = OmegaConf.load("conf/dgp/openml.yaml")
        dgp_cfg.name = dgp_name
    else:
        raise ValueError(f"No DGP config found for {dgp_name}.")
    cfg.dgp = dgp_cfg
    cfg.data_size = int(metadata["data_size"])
    cfg.num_queries = int(metadata["num_queries"])
    cfg.rollout_length = int(metadata["rollout_length"])
    cfg.num_posterior_samples = int(metadata["num_posterior_samples"])
    cfg.seed = int(metadata["seed"])
    return cfg


def load_query_pickle(path: Path) -> pd.DataFrame:
    query_data = utils.read_from(str(path))
    query_source = query_data.get("source", "test_data")
    return pd.DataFrame(
        {
            "query_id": np.asarray(query_data["idx"], dtype=np.int32),
            "y_true": np.asarray(query_data["y"], dtype=object),
            "x_q": list(np.asarray(query_data["x"], dtype=np.float64)),
            "query_source": np.full(len(query_data["idx"]), query_source, dtype=object),
        }
    )


def load_variance_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    var_columns = sorted(
        [col for col in df.columns if col.startswith("var_t")],
        key=lambda col: int(col.replace("var_t", "")),
    )
    if not var_columns:
        raise ValueError(f"No var_t columns found in {path}.")
    return df[["query_id", "y_true", *var_columns]].copy()


def reconstruct_context(cfg, query_df: pd.DataFrame):
    base_key = jax.random.key(cfg.seed * 37)
    base_key, data_key, _, _ = jax.random.split(base_key, 4)
    dgp = load_dgp(cfg, data_key)

    query_source_values = query_df["query_source"].dropna().unique().tolist()
    if len(query_source_values) != 1:
        raise ValueError(f"Expected a single query_source, found {query_source_values}.")
    query_source = query_source_values[0]
    query_idx = query_df["query_id"].to_numpy()

    if query_source == "test_data":
        source_x = np.asarray(dgp.test_data["x"], dtype=np.float64)
        source_y = np.asarray(dgp.test_data["y"], dtype=object)
    elif query_source == "context":
        source_x = np.asarray(dgp.train_data["x"], dtype=np.float64)
        source_y = np.asarray(dgp.train_data["y"], dtype=object)
    else:
        raise ValueError(f"Unsupported query_source={query_source} in saved query pickle.")

    if not np.allclose(np.stack(query_df["x_q"].to_numpy()), source_x[query_idx]):
        raise ValueError("Saved queries do not match reconstructed query source split.")
    if not np.array_equal(query_df["y_true"].to_numpy(dtype=object), source_y[query_idx]):
        raise ValueError("Saved query labels do not match reconstructed query source split.")
    return dgp


def compute_initial_probabilities(cfg, dgp, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_rule = make_classifier_pred_rule(cfg, dgp)
    pred_rule.fit(dgp.train_data["x"], dgp.train_data["y"])
    probabilities = pred_rule.predict_proba(x_query)
    return probabilities, np.asarray(pred_rule.classes_, dtype=object)


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, EPS, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def top1_top2_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def true_class_probability(
    probs: np.ndarray, class_labels: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    class_to_index = {label: idx for idx, label in enumerate(class_labels)}
    true_idx = np.asarray([class_to_index[label] for label in y_true], dtype=np.int32)
    return probs[np.arange(probs.shape[0]), true_idx], true_idx


def compute_knn_features(
    x_query: np.ndarray,
    y_true: np.ndarray,
    x_ctx: np.ndarray,
    y_ctx: np.ndarray,
    k: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for qx, qy in zip(x_query, y_true):
        distances = np.linalg.norm(x_ctx - qx, axis=1)
        sorted_idx = np.argsort(distances)
        knn_idx = sorted_idx[:k]
        knn_dist = distances[knn_idx]
        knn_labels = y_ctx[knn_idx]

        local_counts = pd.Series(knn_labels).value_counts(normalize=True)
        local_probs = np.clip(local_counts.to_numpy(dtype=np.float64), EPS, 1.0)
        local_entropy = float(-(local_probs * np.log(local_probs)).sum())

        same_mask = y_ctx == qy
        diff_mask = ~same_mask
        same_distances = np.sort(distances[same_mask])
        diff_distances = np.sort(distances[diff_mask])
        n_same_available = int(same_distances.size)
        n_diff_available = int(diff_distances.size)
        same_count = min(k, n_same_available)
        diff_count = min(k, n_diff_available)
        same_mean = float(np.mean(same_distances[:same_count])) if same_count > 0 else math.nan
        diff_mean = float(np.mean(diff_distances[:diff_count])) if diff_count > 0 else math.nan

        rows.append(
            {
                "nn_dist": float(knn_dist[0]),
                "knn_avg_dist_k5": float(np.mean(knn_dist)),
                "local_class_entropy_k5": local_entropy,
                "same_diff_gap_k5": diff_mean - same_mean,
                "n_same_available": n_same_available,
                "n_diff_available": n_diff_available,
            }
        )
    return pd.DataFrame(rows)


def fit_loglog_slope(t_values: np.ndarray, variances: np.ndarray) -> float:
    y = np.clip(np.asarray(variances, dtype=np.float64), EPS, None)
    x = np.asarray(t_values, dtype=np.float64)
    slope, _ = np.polyfit(np.log(x), np.log(y), deg=1)
    return float(slope)


def compute_decay_features(variance_df: pd.DataFrame) -> pd.DataFrame:
    var_cols = sorted(
        [col for col in variance_df.columns if col.startswith("var_t")],
        key=lambda col: int(col.replace("var_t", "")),
    )
    if "var_t1" not in variance_df.columns or "var_t100" not in variance_df.columns:
        raise ValueError("Expected var_t1 ... var_t100 columns in variance CSV.")

    records: list[dict[str, float]] = []
    t10 = np.arange(1, 11, dtype=np.float64)
    t20 = np.arange(1, 21, dtype=np.float64)
    t30 = np.arange(1, 31, dtype=np.float64)
    t50 = np.arange(1, 51, dtype=np.float64)
    t100 = np.arange(1, 101, dtype=np.float64)
    for _, row in variance_df.iterrows():
        var_1_to_10 = row[[f"var_t{t}" for t in range(1, 11)]].to_numpy(dtype=np.float64)
        var_1_to_20 = row[[f"var_t{t}" for t in range(1, 21)]].to_numpy(dtype=np.float64)
        var_1_to_30 = row[[f"var_t{t}" for t in range(1, 31)]].to_numpy(dtype=np.float64)
        var_1_to_50 = row[[f"var_t{t}" for t in range(1, 51)]].to_numpy(dtype=np.float64)
        var_1_to_100 = row[[f"var_t{t}" for t in range(1, 101)]].to_numpy(dtype=np.float64)
        records.append(
            {
                "init_variance": float(row["var_t1"]),
                "slope_t_le_10": fit_loglog_slope(t10, var_1_to_10),
                "slope_t_le_20": fit_loglog_slope(t20, var_1_to_20),
                "slope_t_le_30": fit_loglog_slope(t30, var_1_to_30),
                "slope_t_le_50": fit_loglog_slope(t50, var_1_to_50),
                "slope_t_le_100": fit_loglog_slope(t100, var_1_to_100),
            }
        )
    return pd.DataFrame(records)


def compute_correlation_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for target in [
        "slope_t_le_10",
        "slope_t_le_20",
        "slope_t_le_30",
        "slope_t_le_50",
        "slope_t_le_100",
    ]:
        y = summary_df[target].to_numpy(dtype=np.float64)
        for feature in FEATURE_COLUMNS:
            x = summary_df[feature].to_numpy(dtype=np.float64)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            rows.append(
                {
                    "feature_name": feature,
                    "target_slope": target,
                    "pearson_r": float(pearson.statistic),
                    "pearson_pvalue": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_pvalue": float(spearman.pvalue),
                }
            )
    return pd.DataFrame(rows)


def plot_scatter_grid(
    summary_df: pd.DataFrame,
    target: str,
    output_path: Path,
) -> None:
    feature_labels = {
        "init_entropy": "Initial predictive entropy",
        "local_class_entropy_k5": "Local class entropy (k=5)",
        "knn_avg_dist_k5": "k-NN average distance (k=5)",
        "init_margin_top1_top2": "Initial top1-top2 margin",
        "init_true_class_prob": "Initial true-class probability",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = np.asarray(axes).reshape(-1)
    y = summary_df[target].to_numpy(dtype=np.float64)
    highlight_mask = summary_df["query_id"].isin(HIGHLIGHT_QUERY_IDS).to_numpy()
    base_mask = ~highlight_mask

    for ax, feature in zip(axes, SCATTER_FEATURE_COLUMNS):
        x = summary_df[feature].to_numpy(dtype=np.float64)
        ax.scatter(x[base_mask], y[base_mask], color="#94a3b8", alpha=0.8, s=38)
        for qid in HIGHLIGHT_QUERY_IDS:
            qmask = summary_df["query_id"].to_numpy() == qid
            if not np.any(qmask):
                continue
            ax.scatter(
                x[qmask],
                y[qmask],
                color=HIGHLIGHT_COLORS[qid],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.95,
                s=58,
                zorder=3,
            )
        pearson = pearsonr(x, y).statistic
        spearman = spearmanr(x, y).statistic
        ax.set_xlabel(feature_labels[feature])
        ax.set_ylabel(f"{target} (log-log)")
        ax.set_title(f"{feature_labels[feature]} vs {target}")
        ax.text(
            0.03,
            0.97,
            f"Pearson r={pearson:.2f}\nSpearman rho={spearman:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.grid(alpha=0.25)

    for ax in axes[len(SCATTER_FEATURE_COLUMNS) :]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=HIGHLIGHT_COLORS[qid],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=7,
            label=f"q={qid}",
        )
        for qid in HIGHLIGHT_QUERY_IDS
    ]
    fig.suptitle(f"Query Features vs Variance Decay: {target}", y=0.995)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rank_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    ranked = summary_df.sort_values("slope_t_le_50", ascending=True).reset_index(drop=True)
    y_pos = np.arange(ranked.shape[0])
    color_values = ranked["local_class_entropy_k5"].to_numpy(dtype=np.float64)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        y_pos,
        ranked["slope_t_le_50"].to_numpy(dtype=np.float64),
        color=cmap(norm(color_values)),
        alpha=0.9,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(qid) for qid in ranked["query_id"]])
    ax.invert_yaxis()
    ax.set_xlabel("Overall log-log slope of variance (t ≤ 50)")
    ax.set_ylabel("Query ID")
    ax.set_title("Ranked Query Variance Decay Slopes")
    ax.grid(axis="x", alpha=0.25)

    for bar, entropy in zip(bars, color_values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2.0,
            f"  H_local={entropy:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Local class entropy (k=5)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_short_summary(
    summary_df: pd.DataFrame,
    correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    top50 = (
        correlations[correlations["target_slope"] == "slope_t_le_50"]
        .assign(abs_spearman=lambda df: df["spearman_rho"].abs())
        .sort_values("abs_spearman", ascending=False)
    )
    top100 = (
        correlations[correlations["target_slope"] == "slope_t_le_100"]
        .assign(abs_spearman=lambda df: df["spearman_rho"].abs())
        .sort_values("abs_spearman", ascending=False)
    )
    fastest = summary_df.sort_values("slope_t_le_50").head(5)[
        ["query_id", "slope_t_le_50", "init_entropy", "local_class_entropy_k5", "knn_avg_dist_k5"]
    ]
    slowest = summary_df.sort_values("slope_t_le_50", ascending=False).head(5)[
        ["query_id", "slope_t_le_50", "init_entropy", "local_class_entropy_k5", "knn_avg_dist_k5"]
    ]

    def frame_to_bullets(frame: pd.DataFrame) -> list[str]:
        lines: list[str] = []
        for _, row in frame.iterrows():
            lines.append(
                "- "
                f"q={int(row['query_id'])}, "
                f"slope_t_le_50={row['slope_t_le_50']:.3f}, "
                f"init_entropy={row['init_entropy']:.3f}, "
                f"local_class_entropy_k5={row['local_class_entropy_k5']:.3f}, "
                f"knn_avg_dist_k5={row['knn_avg_dist_k5']:.3f}"
            )
        return lines

    lines = [
        "# Query Property vs Variance Decay",
        "",
        "This analysis reused the saved 24-query pickle and reconstructed the original fixed context D_0 using the same dgp/data_size/seed configuration as the conditional variance run. The saved queries were verified against the reconstructed test split before analysis.",
        "",
        "## Strongest feature associations with slope_t_le_50 (by |Spearman rho|)",
    ]
    for _, row in top50.head(3).iterrows():
        lines.append(
            f"- `{row['feature_name']}`: Pearson r={row['pearson_r']:.3f}, Spearman rho={row['spearman_rho']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Strongest feature associations with slope_t_le_100 (by |Spearman rho|)",
        ]
    )
    for _, row in top100.head(3).iterrows():
        lines.append(
            f"- `{row['feature_name']}`: Pearson r={row['pearson_r']:.3f}, Spearman rho={row['spearman_rho']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Fastest-decay queries (most negative slope_t_le_50)",
            *frame_to_bullets(fastest),
            "",
            "## Slowest-decay queries (most positive slope_t_le_50)",
            *frame_to_bullets(slowest),
            "",
            "Interpretation should remain exploratory because n=24 is small and the analysis is based on one fixed context.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(output_dir: Path, k: int) -> None:
    metadata = parse_output_metadata(output_dir)
    cfg = load_analysis_cfg(metadata)

    query_df = load_query_pickle(output_dir / "queries.pickle")
    variance_df = load_variance_csv(output_dir / "conditional-theta-variance.csv")
    summary_df = variance_df[["query_id", "y_true"]].merge(query_df, on=["query_id", "y_true"], how="inner")
    if summary_df.shape[0] != int(metadata["num_queries"]):
        raise ValueError("Query pickle and variance CSV could not be aligned cleanly.")

    dgp = reconstruct_context(cfg, summary_df)
    x_query = np.stack(summary_df["x_q"].to_numpy())
    y_true = summary_df["y_true"].to_numpy(dtype=object)
    x_ctx = np.asarray(dgp.train_data["x"], dtype=np.float64)
    y_ctx = np.asarray(dgp.train_data["y"], dtype=object)

    probs0, class_labels = compute_initial_probabilities(cfg, dgp, x_query)
    init_true_prob, true_class_idx = true_class_probability(probs0, class_labels, y_true)
    query_feature_df = pd.DataFrame(
        {
            "init_entropy": entropy_from_probs(probs0),
            "init_margin_top1_top2": top1_top2_margin(probs0),
            "init_true_class_prob": init_true_prob,
            "predicted_class": class_labels[np.argmax(probs0, axis=1)],
            "true_class_idx": true_class_idx,
            "k_used": k,
        }
    )
    knn_df = compute_knn_features(x_query, y_true, x_ctx, y_ctx, k=k)
    decay_df = compute_decay_features(variance_df)

    output_summary = pd.concat(
        [
            summary_df[["query_id", "y_true"]],
            query_feature_df,
            knn_df,
            decay_df,
        ],
        axis=1,
    )

    summary_path = output_dir / "query_feature_vs_variance_decay_summary.csv"
    output_summary.to_csv(summary_path, index=False)

    ranked_path = output_dir / "query_feature_vs_variance_decay_ranked_by_slope_t_le_50.csv"
    output_summary.sort_values("slope_t_le_50", ascending=True).to_csv(ranked_path, index=False)

    correlations = compute_correlation_rows(output_summary)
    correlations.to_csv(output_dir / "query_feature_vs_variance_decay_correlations.csv", index=False)

    plot_scatter_grid(
        output_summary,
        "slope_t_le_10",
        output_dir / "query_feature_vs_variance_decay_scatter_slope_t_le_10.png",
    )
    plot_scatter_grid(
        output_summary,
        "slope_t_le_20",
        output_dir / "query_feature_vs_variance_decay_scatter_slope_t_le_20.png",
    )
    plot_scatter_grid(
        output_summary,
        "slope_t_le_30",
        output_dir / "query_feature_vs_variance_decay_scatter_slope_t_le_30.png",
    )
    plot_scatter_grid(
        output_summary,
        "slope_t_le_50",
        output_dir / "query_feature_vs_variance_decay_scatter_slope_t_le_50.png",
    )
    plot_scatter_grid(
        output_summary,
        "slope_t_le_100",
        output_dir / "query_feature_vs_variance_decay_scatter_slope_t_le_100.png",
    )
    plot_rank_comparison(
        output_summary,
        output_dir / "query_feature_vs_variance_decay_rank_plot.png",
    )
    write_short_summary(
        output_summary,
        correlations,
        output_dir / "query_feature_vs_variance_decay_summary_note.md",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze fixed-query properties against variance decay slopes."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    run_analysis(args.output_dir.resolve(), k=args.k)


if __name__ == "__main__":
    main()
