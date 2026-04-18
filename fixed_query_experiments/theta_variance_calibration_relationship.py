from __future__ import annotations

import csv
import logging
import os
import warnings

import hydra
import jax
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import trange

from dgp import load_dgp
from fixed_query_experiments import metrics
from fixed_query_experiments.rollout import (
    collect_one_step_conditional_beliefs,
    make_classifier_pred_rule,
    sample_context_queries,
    sample_reference_trajectory,
)
import utils

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def run(cfg: DictConfig) -> None:
    if cfg.query_source != "context":
        raise ValueError(
            "Theta-variance calibration relationship experiment requires "
            "query_source='context'."
        )

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)

    split_rows: list[dict[str, float | int]] = []
    split_query_rows: list[dict[str, float | int | str]] = []
    details_path = f"{outdir}/split_level.csv"
    query_details_path = f"{outdir}/split_query_level.csv"
    query_records: list[dict[str, object]] = []

    slope_windows = [int(window) for window in cfg.slope_windows]
    if max(slope_windows) > cfg.rollout_length:
        raise ValueError(
            f"All slope windows must be <= rollout_length={cfg.rollout_length}, got {slope_windows}."
        )

    for repeat_idx in trange(cfg.n_repeats, desc="Repeats"):
        repeat_key = jax.random.fold_in(base_key, repeat_idx)
        data_key, reference_key, continuation_key = jax.random.split(repeat_key, 3)
        dgp = load_dgp(cfg, data_key)

        query_idx, x_query, y_query = sample_context_queries(
            dgp.train_data,
            cfg.num_queries,
            cfg.seed * 101 + repeat_idx,
        )

        pred_rule = make_classifier_pred_rule(cfg, dgp)
        pred_rule.fit(dgp.train_data["x"], dgp.train_data["y"])
        query_probs = pred_rule.predict_proba(x_query)
        class_labels = np.asarray(pred_rule.classes_)
        calibration = metrics.compute_calibration_metrics(
            query_probs,
            class_labels,
            y_query,
            ece_bins=cfg.ece_bins,
        )

        x_reference, y_reference = sample_reference_trajectory(cfg, dgp, reference_key)
        conditional_beliefs = collect_one_step_conditional_beliefs(
            cfg,
            dgp,
            continuation_key,
            x_reference,
            y_reference,
            x_query,
        )
        theta_samples, _ = metrics.select_true_label_probabilities(
            conditional_beliefs,
            class_labels,
            y_query,
        )
        variance_summary = metrics.compute_query_variance_summary(
            theta_samples,
            depths=np.arange(cfg.rollout_length + 1, dtype=np.int32),
            ddof=cfg.conditional_theta_variance.ddof,
        )
        slopes_by_window = metrics.compute_query_loglog_slopes(
            variance_summary["variance_by_query"],
            variance_summary["depths"],
            slope_windows,
            eps=cfg.slope_eps,
        )

        split_row: dict[str, float | int] = {
            "repeat_idx": repeat_idx,
            "acc": float(calibration["accuracy"]),
            "nll": float(calibration["nll"]),
            "ece": float(calibration["ece"]),
        }
        for window in slope_windows:
            slopes = slopes_by_window[window]
            split_row[f"mean_slope_{window}"] = float(np.mean(slopes))
            split_row[f"median_slope_{window}"] = float(np.median(slopes))
            split_row[f"q25_slope_{window}"] = float(np.quantile(slopes, 0.25))
            split_row[f"q75_slope_{window}"] = float(np.quantile(slopes, 0.75))
        split_rows.append(split_row)

        variance_by_query = variance_summary["variance_by_query"]
        for query_pos in range(cfg.num_queries):
            row: dict[str, float | int | str] = {
                "repeat_idx": repeat_idx,
                "query_id": int(query_idx[query_pos]),
                "y_true": y_query[query_pos],
            }
            for t in range(1, cfg.rollout_length + 1):
                row[f"var_{t}"] = float(variance_by_query[query_pos, t])
            for window in slope_windows:
                row[f"slope_{window}"] = float(slopes_by_window[window][query_pos])
            split_query_rows.append(row)

        query_records.append(
            {
                "repeat_idx": repeat_idx,
                "idx": np.asarray(query_idx, dtype=np.int32),
                "x": np.asarray(x_query, dtype=np.float64),
                "y": np.asarray(y_query),
                "source": cfg.query_source,
            }
        )

    pd.DataFrame(split_rows).to_csv(details_path, index=False)
    pd.DataFrame(split_query_rows).to_csv(query_details_path, index=False)
    utils.write_to(f"{outdir}/queries_by_split.pickle", query_records, verbose=True)
    utils.write_to(
        f"{outdir}/config_snapshot.pickle",
        {
            "dgp_name": cfg.dgp.name,
            "data_size": cfg.data_size,
            "num_queries": cfg.num_queries,
            "query_source": cfg.query_source,
            "n_repeats": cfg.n_repeats,
            "rollout_length": cfg.rollout_length,
            "num_posterior_samples": cfg.num_posterior_samples,
            "ece_bins": cfg.ece_bins,
            "slope_windows": slope_windows,
            "slope_eps": cfg.slope_eps,
        },
        verbose=True,
    )
    logging.info("Wrote split-level CSV to %s", details_path)
    logging.info("Wrote split-query-level CSV to %s", query_details_path)
