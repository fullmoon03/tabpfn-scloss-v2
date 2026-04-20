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
from fixed_query_experiments.plotting import plot_mean_belief
from fixed_query_experiments.rollout import (
    collect_rollout_beliefs,
    sample_context_queries,
    sample_test_queries,
)
import utils

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)

def run(cfg: DictConfig) -> None:
    outdir = os.path.relpath(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)
    base_key, data_key = jax.random.split(base_key)

    dgp = load_dgp(cfg, data_key)
    if cfg.query_source == "test_data":
        if getattr(dgp, "test_data", None) is None:
            raise ValueError(
                f"{cfg.dgp.name} does not provide a held-out test split for query sampling."
            )
        query_idx, x_query, y_query = sample_test_queries(
            dgp.test_data, cfg.num_queries, cfg.seed * 101
        )
    elif cfg.query_source == "context":
        query_idx, x_query, y_query = sample_context_queries(
            dgp.train_data, cfg.num_queries, cfg.seed * 101
        )
    else:
        raise ValueError(
            f"Unsupported query_source={cfg.query_source}. Use 'test_data' or 'context'."
        )

    utils.write_to(
        f"{outdir}/queries.pickle",
        {"idx": query_idx, "x": x_query, "y": y_query, "source": cfg.query_source},
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
            "query_source": cfg.query_source,
            "class_labels": class_labels,
        },
        verbose=True,
    )
    if cfg.metrics.emd.enabled:
        emd_metrics = metrics.compute_expected_martingale_drift(
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
