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
from fixed_query_experiments.rollout import (
    collect_one_step_conditional_beliefs,
    make_classifier_pred_rule,
    sample_context_queries,
    sample_reference_trajectory,
    sample_test_queries,
)
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
    base_key, data_key, reference_key, continuation_key = jax.random.split(base_key, 4)

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

    x_reference, y_reference = sample_reference_trajectory(cfg, dgp, reference_key)
    utils.write_to(
        f"{outdir}/reference-trajectory.pickle",
        {"x": x_reference, "y": y_reference},
        verbose=True,
    )

    conditional_beliefs = collect_one_step_conditional_beliefs(
        cfg,
        dgp,
        continuation_key,
        x_reference,
        y_reference,
        x_query,
    )
    pred_rule = make_classifier_pred_rule(cfg, dgp)
    pred_rule.fit(dgp.train_data["x"], dgp.train_data["y"])
    class_labels = np.asarray(pred_rule.classes_)

    theta_samples, true_class_idx = metrics.select_true_label_probabilities(
        conditional_beliefs,
        class_labels,
        y_query,
    )
    theta_by_depth = theta_samples.mean(axis=1)
    variance_summary = metrics.compute_query_variance_summary(
        theta_samples,
        depths=np.arange(cfg.rollout_length + 1, dtype=np.int32),
        ddof=cfg.conditional_theta_variance.ddof,
    )

    output = {
        "metric_name": "conditional_true_label_probability_variance",
        "belief_type": "one_step_conditional",
        "note": (
            "For each reference depth t, sample one-step continuations z_{t+1}^{(m)} from D_t "
            "and evaluate the true-label belief under D_t union {z_{t+1}^{(m)}}."
        ),
        "theta_by_depth": theta_by_depth,
        "theta_samples": theta_samples,
        "conditional_beliefs": conditional_beliefs,
        "query_idx": query_idx,
        "x_query": x_query,
        "y_query": y_query,
        "query_source": cfg.query_source,
        "class_labels": class_labels,
        "true_class_idx": true_class_idx,
        "num_posterior_samples": cfg.num_posterior_samples,
        "rollout_length": cfg.rollout_length,
        **variance_summary,
    }
    utils.write_to(
        f"{outdir}/conditional-theta-variance.pickle",
        output,
        verbose=True,
    )

    write_query_metric_csv(
        f"{outdir}/conditional-theta-variance.csv",
        "var_t",
        query_idx,
        y_query,
        variance_summary["variance_by_query"],
        start_index=0,
    )
    write_query_summary_csv(
        f"{outdir}/conditional-theta-variance-summary.csv",
        query_idx,
        y_query,
        variance_summary["peak_step"],
        variance_summary["max_variance"],
    )
    logging.info(
        "Mean max conditional theta variance across queries: %.6f",
        float(np.mean(variance_summary["max_variance"])),
    )
