import logging
import os
import warnings

import hydra
import jax
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import belief_metrics
from fixed_query_rollout import (
    collect_rollout_beliefs,
    sample_test_queries,
)
import utils
from dgp import load_dgp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def plot_mean_belief(
    mean_belief: np.ndarray,
    std_belief: np.ndarray,
    query_indices: np.ndarray,
    class_labels: np.ndarray,
    queries_per_figure: int,
    plot_rows: int,
    plot_cols: int,
    outdir: str,
    dataset_name: str,
) -> None:
    depth = np.arange(mean_belief.shape[0])
    num_queries = mean_belief.shape[1]
    num_classes = mean_belief.shape[2]
    class_handles = None

    for fig_idx, start in enumerate(range(0, num_queries, queries_per_figure), start=1):
        end = min(start + queries_per_figure, num_queries)
        fig, axes = plt.subplots(
            plot_rows, plot_cols, figsize=(plot_cols * 4.0, plot_rows * 3.0), sharex=True, sharey=True
        )
        axes = np.asarray(axes).reshape(-1)

        for ax_idx, query_pos in enumerate(range(start, end)):
            ax = axes[ax_idx]
            line_handles = []
            for class_idx in range(num_classes):
                (line,) = ax.plot(
                    depth,
                    mean_belief[:, query_pos, class_idx],
                    linewidth=1.5,
                    marker="o",
                    markersize=2.2,
                    label=f"class {class_idx} ({class_labels[class_idx]})",
                )
                lower = np.clip(
                    mean_belief[:, query_pos, class_idx] - std_belief[:, query_pos, class_idx],
                    0.0,
                    1.0,
                )
                upper = np.clip(
                    mean_belief[:, query_pos, class_idx] + std_belief[:, query_pos, class_idx],
                    0.0,
                    1.0,
                )
                ax.fill_between(depth, lower, upper, color=line.get_color(), alpha=0.18)
                line_handles.append(line)
            if class_handles is None:
                class_handles = line_handles

            ax.set_title(f"q_idx={query_indices[query_pos]}")
            ax.set_xlim(depth[0], depth[-1])
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)

        for ax in axes[end - start :]:
            ax.axis("off")

        fig.suptitle(
            f"{dataset_name.capitalize()} fixed-query mean belief across paths "
            f"({start + 1}-{end}/{num_queries})",
            y=0.995,
        )
        fig.supxlabel("Depth n")
        fig.supylabel("Mean Belief")
        if class_handles is not None:
            fig.legend(
                class_handles,
                [handle.get_label() for handle in class_handles],
                loc="upper right",
                frameon=True,
            )
        fig.tight_layout(rect=(0, 0, 0.97, 0.97))
        fig.savefig(f"{outdir}/belief-trajectory-{fig_idx}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="fixed-query-belief")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    outdir = os.path.relpath(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)
    base_key, data_key = jax.random.split(base_key)

    dgp = load_dgp(cfg, data_key)
    if getattr(dgp, "test_data", None) is None:
        raise ValueError(f"{cfg.dgp.name} does not provide a held-out test split for query sampling.")

    query_idx, x_query, y_query = sample_test_queries(dgp.test_data, cfg.num_queries, cfg.seed * 101)
    utils.write_to(
        f"{outdir}/queries.pickle",
        {"idx": query_idx, "x": x_query, "y": y_query},
        verbose=True,
    )

    beliefs, class_labels = collect_rollout_beliefs(
        cfg,
        dgp,
        base_key,
        x_query,
        progress_desc="Rollouts",
    )

    mean_belief = beliefs.mean(axis=0)
    std_belief = beliefs.std(axis=0)
    utils.write_to(
        f"{outdir}/belief-trajectory.pickle",
        {
            "beliefs": beliefs,
            "mean_belief": mean_belief,
            "std_belief": std_belief,
            "query_idx": query_idx,
            "x_query": x_query,
            "y_query": y_query,
            "class_labels": class_labels,
        },
        verbose=True,
    )
    if cfg.metrics.emd.enabled:
        emd_metrics = belief_metrics.compute_expected_martingale_drift(
            mean_belief,
            distance_name=cfg.metrics.emd.distance,
            reference_depth=cfg.metrics.emd.reference_depth,
            average_from_depth=cfg.metrics.emd.average_from_depth,
        )
        emd_metrics["query_idx"] = query_idx
        emd_metrics["x_query"] = x_query
        emd_metrics["y_query"] = y_query
        utils.write_to(
            f"{outdir}/metrics/expected-martingale-drift.pickle",
            emd_metrics,
            verbose=True,
        )
        logging.info(
            "Global EMD (%s): %.6f",
            cfg.metrics.emd.distance,
            emd_metrics["global_emd"],
        )
    plot_mean_belief(
        mean_belief,
        std_belief,
        query_idx,
        class_labels,
        cfg.queries_per_figure,
        cfg.plot_rows,
        cfg.plot_cols,
        outdir,
        cfg.dgp.name,
    )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
