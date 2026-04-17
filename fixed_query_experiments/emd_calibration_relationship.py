from __future__ import annotations

import csv
import logging
import os
import warnings

import hydra
import jax
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import trange

from analysis.emd_calibration_relationship_plots import run_from_details_csv
from dgp import load_dgp
from fixed_query_experiments import metrics
from fixed_query_experiments.rollout import (
    collect_rollout_beliefs,
    make_classifier_pred_rule,
    sample_context_queries,
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
            "EMD-calibration relationship experiment currently requires "
            "query_source='context'."
        )

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)
    details_path = f"{outdir}/details.csv"

    fieldnames = ["repeat_id", "EMD", "NLL", "ECE", "Accuracy"]
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for repeat_idx in trange(cfg.n_repeats, desc="Repeats"):
            repeat_key = jax.random.fold_in(base_key, repeat_idx)
            data_key, rollout_key = jax.random.split(repeat_key)
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

            beliefs, _ = collect_rollout_beliefs(
                cfg,
                dgp,
                rollout_key,
                x_query,
                progress_desc=f"Repeat {repeat_idx + 1}/{cfg.n_repeats} rollouts",
            )
            mean_belief = beliefs.mean(axis=0)
            emd_metrics = metrics.compute_expected_martingale_drift(
                mean_belief,
                distance_name=cfg.metrics.emd.distance,
                reference_depth=cfg.metrics.emd.reference_depth,
                average_from_depth=cfg.metrics.emd.average_from_depth,
            )

            writer.writerow(
                {
                    "repeat_id": repeat_idx,
                    "EMD": float(emd_metrics["global_emd"]),
                    "NLL": float(calibration["nll"]),
                    "ECE": float(calibration["ece"]),
                    "Accuracy": float(calibration["accuracy"]),
                }
            )
            f.flush()

    run_from_details_csv(details_path)
    utils.write_to(
        f"{outdir}/config_snapshot.pickle",
        {
            "dgp_name": cfg.dgp.name,
            "data_size": cfg.data_size,
            "num_queries": cfg.num_queries,
            "query_source": cfg.query_source,
            "n_repeats": cfg.n_repeats,
            "rollout_times": cfg.rollout_times,
            "rollout_length": cfg.rollout_length,
            "ece_bins": cfg.ece_bins,
        },
        verbose=True,
    )
    logging.info("Finished EMD-calibration experiment. Details saved to %s", details_path)
