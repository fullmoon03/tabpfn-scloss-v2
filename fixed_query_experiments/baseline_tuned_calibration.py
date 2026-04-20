from __future__ import annotations

import csv
import logging
import os
import warnings

import hydra
import jax
import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import trange

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgp import OPENML_BINARY_CLASSIFICATION, OPENML_CLASSIFICATION, load_dgp
from fixed_query_experiments import metrics
from fixed_query_experiments.rollout import make_classifier_pred_rule
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


def _align_probabilities(
    probabilities: np.ndarray,
    source_classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    source_classes = np.asarray(source_classes)
    target_classes = np.asarray(target_classes)
    aligned = np.zeros((probabilities.shape[0], target_classes.shape[0]), dtype=np.float64)
    for source_idx, label in enumerate(source_classes):
        matches = np.flatnonzero(target_classes == label)
        if matches.size != 1:
            raise ValueError(f"Class label {label!r} is missing from target class labels.")
        aligned[:, matches[0]] = probabilities[:, source_idx]
    return aligned


def _make_tuned_pred_rule(cfg: DictConfig, dgp: object, class_labels: np.ndarray):
    if cfg.dgp.name in OPENML_CLASSIFICATION + OPENML_BINARY_CLASSIFICATION:
        if any(getattr(dgp, "categorical_x", [])):
            raise NotImplementedError(
                "Fine-tuned TabPFN2 calibration comparison currently treats all "
                "features as numeric; categorical OpenML features are not wired in."
            )
    return FineTunedTabPFNClassifierPredRule(
        base_checkpoint_path=cfg.base_checkpoint_path,
        tuned_checkpoint_path=cfg.tuned_checkpoint_path,
        n_num_features=dgp.train_data["x"].shape[1],
        class_labels=class_labels,
        device=cfg.device,
        torch_seed=cfg.seed * 131,
    )


def _evaluate_pred_rule(
    *,
    pred_rule,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    class_labels: np.ndarray,
    ece_bins: int,
) -> dict[str, float]:
    pred_rule.fit(x_context, y_context)
    probabilities = pred_rule.predict_proba(x_eval)
    probabilities = _align_probabilities(probabilities, pred_rule.classes_, class_labels)
    return metrics.compute_calibration_metrics(
        probabilities,
        class_labels,
        y_eval,
        ece_bins=ece_bins,
    )


def _plot_ece_nll(details_path: str, output_path: str) -> None:
    df = pd.read_csv(details_path)
    colors = {"baseline": "#2563eb", "tuned": "#dc2626"}

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    for model_name, group in df.groupby("model", sort=False):
        ax.scatter(
            group["ece"],
            group["nll"],
            s=30,
            alpha=0.75,
            label=model_name,
            color=colors.get(model_name, None),
            edgecolor="white",
            linewidth=0.35,
        )
    ax.set_xlabel("ECE")
    ax.set_ylabel("NLL")
    ax.set_title(
        "Baseline vs. Tuned TabPFN: ECE-NLL Comparison over "
        f"{df['repeat_id'].nunique()} Repeated Splits"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(cfg: DictConfig) -> None:
    if not cfg.tuned_checkpoint_path:
        raise ValueError("Provide tuned_checkpoint_path=/path/to/best_checkpoint.pt.")

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)
    details_path = f"{outdir}/baseline_vs_tuned_ece_nll.csv"
    fieldnames = ["repeat_id", "model", "nll", "ece", "accuracy", "n_eval"]

    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        tuned_pred_rule = None
        tuned_class_labels = None

        for repeat_idx in trange(cfg.n_repeats, desc="Repeats"):
            data_key = jax.random.fold_in(base_key, repeat_idx)
            dgp = load_dgp(cfg, data_key)
            if getattr(dgp, "test_data", None) is None:
                raise ValueError(f"{cfg.dgp.name} does not provide an evaluation split.")

            x_context = dgp.train_data["x"]
            y_context = dgp.train_data["y"]
            x_eval = dgp.test_data["x"]
            y_eval = dgp.test_data["y"]
            class_labels = _class_labels(dgp)

            baseline_pred_rule = make_classifier_pred_rule(cfg, dgp)
            baseline_metrics = _evaluate_pred_rule(
                pred_rule=baseline_pred_rule,
                x_context=x_context,
                y_context=y_context,
                x_eval=x_eval,
                y_eval=y_eval,
                class_labels=class_labels,
                ece_bins=cfg.ece_bins,
            )
            if tuned_pred_rule is None:
                tuned_pred_rule = _make_tuned_pred_rule(cfg, dgp, class_labels)
                tuned_class_labels = class_labels
            elif not np.array_equal(tuned_class_labels, class_labels):
                raise ValueError(
                    f"Class labels changed across repeats: {tuned_class_labels} vs {class_labels}"
                )
            tuned_metrics = _evaluate_pred_rule(
                pred_rule=tuned_pred_rule,
                x_context=x_context,
                y_context=y_context,
                x_eval=x_eval,
                y_eval=y_eval,
                class_labels=class_labels,
                ece_bins=cfg.ece_bins,
            )

            for model_name, result in [
                ("baseline", baseline_metrics),
                ("tuned", tuned_metrics),
            ]:
                writer.writerow(
                    {
                        "repeat_id": repeat_idx,
                        "model": model_name,
                        "nll": float(result["nll"]),
                        "ece": float(result["ece"]),
                        "accuracy": float(result["accuracy"]),
                        "n_eval": int(x_eval.shape[0]),
                    }
                )
            f.flush()

    plot_path = f"{outdir}/baseline_vs_tuned_ece_nll_scatter.png"
    _plot_ece_nll(details_path, plot_path)
    utils.write_to(
        f"{outdir}/config_snapshot.pickle",
        {
            "dgp_name": cfg.dgp.name,
            "context_size": cfg.data_size,
            "n_repeats": cfg.n_repeats,
            "ece_bins": cfg.ece_bins,
            "tuned_checkpoint_path": cfg.tuned_checkpoint_path,
        },
        verbose=True,
    )
    logging.info("Finished baseline-vs-tuned calibration comparison: %s", details_path)
