from __future__ import annotations

import csv
import logging
import os
import warnings
from types import SimpleNamespace

import hydra
import jax
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from analysis.context_size_effect_plots import run_from_details_csv
from dgp import load_dgp
from fixed_query_experiments import metrics
from fixed_query_experiments.rollout import (
    collect_rollout_beliefs,
    make_classifier_pred_rule,
    sample_queries,
)
import utils

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def _make_context_dgp(base_dgp: object, x_context: np.ndarray, y_context: np.ndarray) -> object:
    return SimpleNamespace(
        train_data={"x": x_context, "y": y_context},
        categorical_x=getattr(base_dgp, "categorical_x", None),
    )


def _sample_fixed_queries(
    full_data: dict[str, np.ndarray], num_queries: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sample_queries(full_data, num_queries, seed)


def _sample_context_indices(
    available_idx: np.ndarray,
    context_size: int,
    context_seed: int,
    size_idx: int,
    repeat_idx: int,
) -> np.ndarray:
    seed = context_seed + 1009 * size_idx + repeat_idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(available_idx, size=context_size, replace=False))


def _align_probabilities_to_class_space(
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    target_class_labels: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    class_labels = np.asarray(class_labels)
    target_class_labels = np.asarray(target_class_labels)

    aligned = np.zeros(
        (probabilities.shape[0], target_class_labels.shape[0]), dtype=np.float64
    )
    source_index = {label: idx for idx, label in enumerate(class_labels)}
    for target_idx, label in enumerate(target_class_labels):
        source_idx = source_index.get(label)
        if source_idx is not None:
            aligned[:, target_idx] = probabilities[:, source_idx]
    return aligned


def run(cfg: DictConfig) -> None:
    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    data_key = jax.random.key(cfg.seed * 37)
    base_dgp = load_dgp(cfg, data_key)

    if not hasattr(base_dgp, "full_data"):
        raise ValueError(
            f"{cfg.dgp.name} does not expose full_data, which is required for this experiment."
        )

    full_x = np.asarray(base_dgp.full_data["x"])
    full_y = np.asarray(base_dgp.full_data["y"])
    global_class_labels = np.unique(full_y)
    n_total = full_x.shape[0]

    context_sizes = [int(size) for size in cfg.context_sizes]
    if not context_sizes:
        raise ValueError("context_sizes must contain at least one size.")
    if min(context_sizes) <= 0:
        raise ValueError(f"context_sizes must be positive, got {context_sizes}.")
    if max(context_sizes) + cfg.num_queries > n_total:
        raise ValueError(
            "Not enough samples to hold fixed queries and largest context without overlap: "
            f"{max(context_sizes)} + {cfg.num_queries} > {n_total}."
        )

    query_idx, x_query, y_query = _sample_fixed_queries(
        {"x": full_x, "y": full_y},
        cfg.num_queries,
        cfg.query_seed,
    )
    remaining_mask = np.ones(n_total, dtype=bool)
    remaining_mask[query_idx] = False
    available_idx = np.flatnonzero(remaining_mask)

    utils.write_to(
        f"{outdir}/queries.pickle",
        {
            "idx": query_idx,
            "x": x_query,
            "y": y_query,
            "source": "full_data",
        },
        verbose=True,
    )

    details_path = f"{outdir}/details.csv"
    context_selections: list[dict[str, object]] = []
    fieldnames = ["context_size", "repeat_idx", "Accuracy", "NLL", "ECE", "EMD"]

    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for size_idx, context_size in enumerate(context_sizes):
            for repeat_idx in tqdm(
                range(cfg.n_repeats),
                desc=f"context_size={context_size}",
            ):
                context_idx = _sample_context_indices(
                    available_idx,
                    context_size,
                    cfg.context_seed,
                    size_idx,
                    repeat_idx,
                )
                x_context = full_x[context_idx]
                y_context = full_y[context_idx]
                context_dgp = _make_context_dgp(base_dgp, x_context, y_context)

                pred_rule = make_classifier_pred_rule(cfg, context_dgp)
                pred_rule.fit(x_context, y_context)
                query_probs = pred_rule.predict_proba(x_query)
                class_labels = np.asarray(pred_rule.classes_)
                query_probs = _align_probabilities_to_class_space(
                    query_probs,
                    class_labels,
                    global_class_labels,
                )
                calibration = metrics.compute_calibration_metrics(
                    query_probs,
                    global_class_labels,
                    y_query,
                    ece_bins=cfg.ece_bins,
                )

                rollout_key = jax.random.key(
                    cfg.seed * 53 + 10007 * size_idx + repeat_idx
                )
                beliefs, _ = collect_rollout_beliefs(
                    cfg,
                    context_dgp,
                    rollout_key,
                    x_query,
                    progress_desc=(
                        f"context_size={context_size} repeat={repeat_idx + 1}/{cfg.n_repeats}"
                    ),
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
                        "context_size": context_size,
                        "repeat_idx": repeat_idx,
                        "Accuracy": float(calibration["accuracy"]),
                        "NLL": float(calibration["nll"]),
                        "ECE": float(calibration["ece"]),
                        "EMD": float(emd_metrics["global_emd"]),
                    }
                )
                f.flush()

                context_selections.append(
                    {
                        "context_size": context_size,
                        "repeat_idx": repeat_idx,
                        "indices": context_idx,
                    }
                )

    utils.write_to(
        f"{outdir}/context-selections.pickle",
        context_selections,
        verbose=True,
    )
    utils.write_to(
        f"{outdir}/config_snapshot.pickle",
        {
            "dgp_name": cfg.dgp.name,
            "num_queries": cfg.num_queries,
            "query_seed": cfg.query_seed,
            "context_seed": cfg.context_seed,
            "context_sizes": context_sizes,
            "n_repeats": cfg.n_repeats,
            "rollout_times": cfg.rollout_times,
            "rollout_length": cfg.rollout_length,
            "ece_bins": cfg.ece_bins,
        },
        verbose=True,
    )

    run_from_details_csv(details_path)
    logging.info(
        "Finished context-size effect experiment. Details saved to %s", details_path
    )
