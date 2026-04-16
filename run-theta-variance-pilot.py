import csv
import logging
import os
import warnings

import hydra
import jax
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import belief_metrics
from fixed_query_rollout import collect_rollout_beliefs, sample_test_queries
import utils
from dgp import load_dgp

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def write_theta_variance_csv(
    path: str,
    query_idx: np.ndarray,
    y_query: np.ndarray,
    variance_by_query: np.ndarray,
) -> None:
    fieldnames = ["query_id", "y_true"] + [
        f"var_k{k}" for k in range(1, variance_by_query.shape[1] + 1)
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(variance_by_query.shape[0]):
            row = {
                "query_id": int(query_idx[row_idx]),
                "y_true": y_query[row_idx],
            }
            for depth_idx, value in enumerate(variance_by_query[row_idx], start=1):
                row[f"var_k{depth_idx}"] = float(value)
            writer.writerow(row)


def write_theta_variance_summary_csv(
    path: str,
    query_idx: np.ndarray,
    y_query: np.ndarray,
    peak_step: np.ndarray,
    max_variance: np.ndarray,
) -> None:
    fieldnames = ["query_id", "y_true", "peak_step", "max_variance"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(query_idx.shape[0]):
            writer.writerow(
                {
                    "query_id": int(query_idx[row_idx]),
                    "y_true": y_query[row_idx],
                    "peak_step": int(peak_step[row_idx]),
                    "max_variance": float(max_variance[row_idx]),
                }
            )


@hydra.main(version_base=None, config_path="conf", config_name="theta-variance-pilot")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    outdir = os.path.relpath(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)
    base_key, data_key = jax.random.split(base_key)

    dgp = load_dgp(cfg, data_key)
    if getattr(dgp, "test_data", None) is None:
        raise ValueError(f"{cfg.dgp.name} does not provide a held-out test split for query sampling.")

    query_idx, x_query, y_query = sample_test_queries(
        dgp.test_data, cfg.num_queries, cfg.seed * 101
    )
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
        progress_desc="Theta rollouts",
    )
    theta_metrics = belief_metrics.compute_true_label_variance_trajectory(
        beliefs,
        class_labels,
        y_query,
        ddof=cfg.theta_variance.ddof,
        start_depth=cfg.theta_variance.start_depth,
    )
    theta_metrics["query_idx"] = query_idx
    theta_metrics["x_query"] = x_query
    theta_metrics["y_query"] = y_query
    theta_metrics["class_labels"] = class_labels
    utils.write_to(
        f"{outdir}/theta-true-probability.pickle",
        theta_metrics,
        verbose=True,
    )

    write_theta_variance_csv(
        f"{outdir}/theta-variance.csv",
        query_idx,
        y_query,
        theta_metrics["variance_by_query"],
    )
    write_theta_variance_summary_csv(
        f"{outdir}/theta-variance-summary.csv",
        query_idx,
        y_query,
        theta_metrics["peak_step"],
        theta_metrics["max_variance"],
    )
    logging.info(
        "Mean max theta variance across queries: %.6f",
        float(np.mean(theta_metrics["max_variance"])),
    )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
