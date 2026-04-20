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

from dgp import load_dgp
from fixed_query_experiments import metrics
from fixed_query_experiments.rollout import (
    collect_one_step_conditional_beliefs_with_factory,
    make_classifier_pred_rule,
    sample_context_queries,
    sample_reference_trajectory_with_factory,
)
from fixed_query_experiments.tuned_pred_rule import FineTunedTabPFNClassifierPredRule
import utils

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def _class_labels(dgp: object) -> np.ndarray:
    labels = [dgp.train_data["y"]]
    if getattr(dgp, "test_data", None) is not None:
        labels.append(dgp.test_data["y"])
    return np.unique(np.concatenate(labels))


def _make_tuned_factory(cfg: DictConfig, dgp: object, class_labels: np.ndarray):
    if any(getattr(dgp, "categorical_x", [])):
        raise NotImplementedError(
            "Fine-tuned TabPFN2 conditional variance comparison currently treats all "
            "features as numeric; categorical features are not wired in."
        )

    def factory() -> FineTunedTabPFNClassifierPredRule:
        return FineTunedTabPFNClassifierPredRule(
            base_checkpoint_path=cfg.base_checkpoint_path,
            tuned_checkpoint_path=cfg.tuned_checkpoint_path,
            n_num_features=dgp.train_data["x"].shape[1],
            class_labels=class_labels,
            device=cfg.device,
            torch_seed=cfg.seed * 131,
        )

    return factory


def _write_variance_csvs(
    *,
    outdir: str,
    model_name: str,
    query_idx: np.ndarray,
    y_query: np.ndarray,
    variance_by_query: np.ndarray,
    peak_step: np.ndarray,
    max_variance: np.ndarray,
) -> None:
    wide_path = f"{outdir}/{model_name}-conditional-theta-variance.csv"
    fieldnames = ["model", "query_id", "y_true"] + [
        f"var_t{idx}" for idx in range(variance_by_query.shape[1])
    ]
    with open(wide_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(query_idx.shape[0]):
            row = {
                "model": model_name,
                "query_id": int(query_idx[row_idx]),
                "y_true": y_query[row_idx],
            }
            for depth_idx, value in enumerate(variance_by_query[row_idx]):
                row[f"var_t{depth_idx}"] = float(value)
            writer.writerow(row)

    summary_path = f"{outdir}/{model_name}-conditional-theta-variance-summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "query_id", "y_true", "peak_step", "max_variance"],
        )
        writer.writeheader()
        for row_idx in range(query_idx.shape[0]):
            writer.writerow(
                {
                    "model": model_name,
                    "query_id": int(query_idx[row_idx]),
                    "y_true": y_query[row_idx],
                    "peak_step": int(peak_step[row_idx]),
                    "max_variance": float(max_variance[row_idx]),
                }
            )


def _evaluate_model(
    *,
    cfg: DictConfig,
    dgp: object,
    model_name: str,
    pred_rule_factory,
    reference_key: jax.Array,
    continuation_key: jax.Array,
    query_idx: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    outdir: str,
) -> dict:
    x_reference, y_reference = sample_reference_trajectory_with_factory(
        pred_rule_factory=pred_rule_factory,
        x_train=dgp.train_data["x"],
        y_train=dgp.train_data["y"],
        rollout_length=cfg.rollout_length,
        key=reference_key,
    )
    conditional_beliefs, class_labels = collect_one_step_conditional_beliefs_with_factory(
        pred_rule_factory=pred_rule_factory,
        x_train=dgp.train_data["x"],
        y_train=dgp.train_data["y"],
        base_key=continuation_key,
        x_reference=x_reference,
        y_reference=y_reference,
        x_query=x_query,
        num_posterior_samples=cfg.num_posterior_samples,
        rollout_length=cfg.rollout_length,
        progress_desc=f"{model_name} reference depths",
    )
    theta_samples, true_class_idx = metrics.select_true_label_probabilities(
        conditional_beliefs,
        class_labels,
        y_query,
    )
    variance_summary = metrics.compute_query_variance_summary(
        theta_samples,
        depths=np.arange(cfg.rollout_length + 1, dtype=np.int32),
        ddof=cfg.conditional_theta_variance.ddof,
    )
    output = {
        "metric_name": "conditional_true_label_probability_variance",
        "model_name": model_name,
        "belief_type": "one_step_conditional",
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
        "x_reference": x_reference,
        "y_reference": y_reference,
        **variance_summary,
    }
    utils.write_to(f"{outdir}/{model_name}-conditional-theta-variance.pickle", output)
    _write_variance_csvs(
        outdir=outdir,
        model_name=model_name,
        query_idx=query_idx,
        y_query=y_query,
        variance_by_query=variance_summary["variance_by_query"],
        peak_step=variance_summary["peak_step"],
        max_variance=variance_summary["max_variance"],
    )
    return output


def _write_combined_csv(outdir: str, outputs: list[dict]) -> None:
    path = f"{outdir}/baseline_vs_tuned_conditional_theta_variance.csv"
    max_depths = outputs[0]["variance_by_query"].shape[1]
    fieldnames = ["model", "query_id", "y_true"] + [f"var_t{idx}" for idx in range(max_depths)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for output in outputs:
            for row_idx in range(output["query_idx"].shape[0]):
                row = {
                    "model": output["model_name"],
                    "query_id": int(output["query_idx"][row_idx]),
                    "y_true": output["y_query"][row_idx],
                }
                for depth_idx, value in enumerate(output["variance_by_query"][row_idx]):
                    row[f"var_t{depth_idx}"] = float(value)
                writer.writerow(row)


def run(cfg: DictConfig) -> None:
    if cfg.query_source != "context":
        raise ValueError("This comparison requires query_source=context.")
    if not cfg.tuned_checkpoint_path:
        raise ValueError("Provide tuned_checkpoint_path=/path/to/best_checkpoint.pt.")

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)
    base_key, data_key, reference_key, continuation_key = jax.random.split(base_key, 4)

    dgp = load_dgp(cfg, data_key)
    query_idx, x_query, y_query = sample_context_queries(
        dgp.train_data, cfg.num_queries, cfg.seed * 101
    )
    utils.write_to(
        f"{outdir}/queries.pickle",
        {"idx": query_idx, "x": x_query, "y": y_query, "source": cfg.query_source},
        verbose=True,
    )

    class_labels = _class_labels(dgp)
    baseline_factory = lambda: make_classifier_pred_rule(cfg, dgp)
    tuned_factory = _make_tuned_factory(cfg, dgp, class_labels)

    baseline_output = _evaluate_model(
        cfg=cfg,
        dgp=dgp,
        model_name="baseline",
        pred_rule_factory=baseline_factory,
        reference_key=reference_key,
        continuation_key=continuation_key,
        query_idx=query_idx,
        x_query=x_query,
        y_query=y_query,
        outdir=outdir,
    )
    tuned_output = _evaluate_model(
        cfg=cfg,
        dgp=dgp,
        model_name="tuned",
        pred_rule_factory=tuned_factory,
        reference_key=reference_key,
        continuation_key=continuation_key,
        query_idx=query_idx,
        x_query=x_query,
        y_query=y_query,
        outdir=outdir,
    )
    _write_combined_csv(outdir, [baseline_output, tuned_output])
    logging.info(
        "Mean max conditional theta variance: baseline=%.6f tuned=%.6f",
        float(np.mean(baseline_output["max_variance"])),
        float(np.mean(tuned_output["max_variance"])),
    )
