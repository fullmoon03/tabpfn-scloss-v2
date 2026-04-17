import logging
import os
import warnings

import hydra
import jax
import numpy as np
import torch
from omegaconf import DictConfig

from dgp import load_dgp
from fixed_query_experiments import metrics
from fixed_query_experiments.io import write_query_metric_csv, write_query_summary_csv
from fixed_query_experiments.rollout import collect_rollout_beliefs, sample_test_queries
import utils

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)

def run(cfg: DictConfig) -> None:
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
    theta_metrics = metrics.compute_true_label_variance_trajectory(
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

    write_query_metric_csv(
        f"{outdir}/theta-variance.csv",
        "var_k",
        query_idx,
        y_query,
        theta_metrics["variance_by_query"],
        start_index=1,
    )
    write_query_summary_csv(
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
