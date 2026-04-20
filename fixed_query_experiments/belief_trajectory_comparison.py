from __future__ import annotations

import os

import hydra
import jax
import numpy as np
import torch
from omegaconf import DictConfig

from dgp import OPENML_BINARY_CLASSIFICATION, OPENML_CLASSIFICATION, load_dgp
from fixed_query_experiments.plotting import plot_mean_belief
from fixed_query_experiments.rollout import (
    collect_rollout_beliefs_with_factory,
    make_classifier_pred_rule,
    sample_context_queries,
    sample_test_queries,
)
from fixed_query_experiments.tuned_pred_rule import FineTunedTabPFNClassifierPredRule
import utils


def _sample_fixed_queries(cfg: DictConfig, dgp: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg.query_source == "context":
        return sample_context_queries(dgp.train_data, cfg.num_queries, cfg.seed * 101)
    if cfg.query_source == "test_data":
        if getattr(dgp, "test_data", None) is None:
            raise ValueError(f"{cfg.dgp.name} does not provide test_data.")
        return sample_test_queries(dgp.test_data, cfg.num_queries, cfg.seed * 101)
    raise ValueError(f"Unsupported query_source={cfg.query_source}.")


def _class_labels(dgp: object) -> np.ndarray:
    labels = [dgp.train_data["y"]]
    if getattr(dgp, "test_data", None) is not None:
        labels.append(dgp.test_data["y"])
    return np.unique(np.concatenate(labels))


def _make_tuned_factory(cfg: DictConfig, dgp: object, class_labels: np.ndarray):
    if cfg.dgp.name in OPENML_CLASSIFICATION + OPENML_BINARY_CLASSIFICATION:
        if any(getattr(dgp, "categorical_x", [])):
            raise NotImplementedError(
                "Fine-tuned TabPFN2 comparison currently treats all features as numeric; "
                "categorical OpenML features are not wired into this adapter."
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


def _write_outputs(
    *,
    outdir: str,
    model_name: str,
    beliefs: np.ndarray,
    query_idx: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    class_labels: np.ndarray,
    cfg: DictConfig,
) -> None:
    mean_belief = beliefs.mean(axis=0)
    std_belief = beliefs.std(axis=0)
    utils.write_to(
        f"{outdir}/{model_name}-belief-trajectory.pickle",
        {
            "beliefs": beliefs,
            "mean_belief": mean_belief,
            "std_belief": std_belief,
            "query_idx": query_idx,
            "x_query": x_query,
            "y_query": y_query,
            "query_source": cfg.query_source,
            "class_labels": class_labels,
            "model_name": model_name,
        },
        verbose=True,
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
        filename_prefix=f"{model_name}_queries",
        title_prefix=f"{cfg.dgp.name.capitalize()} {model_name} fixed-query mean belief",
    )


def run(cfg: DictConfig) -> None:
    if cfg.query_source != "context":
        raise ValueError("This paired comparison is intended to run with query_source=context.")
    if not cfg.tuned_checkpoint_path:
        raise ValueError("Provide tuned_checkpoint_path=/path/to/best_checkpoint.pt.")

    outdir = os.path.relpath(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)
    base_key, data_key = jax.random.split(base_key)
    dgp = load_dgp(cfg, data_key)

    query_idx, x_query, y_query = _sample_fixed_queries(cfg, dgp)
    class_labels = _class_labels(dgp)
    utils.write_to(
        f"{outdir}/queries.pickle",
        {
            "idx": query_idx,
            "x": x_query,
            "y": y_query,
            "source": cfg.query_source,
            "class_labels": class_labels,
        },
        verbose=True,
    )

    baseline_factory = lambda: make_classifier_pred_rule(cfg, dgp)
    tuned_factory = _make_tuned_factory(cfg, dgp, class_labels)

    baseline_beliefs, baseline_class_labels = collect_rollout_beliefs_with_factory(
        pred_rule_factory=baseline_factory,
        x_train=dgp.train_data["x"],
        y_train=dgp.train_data["y"],
        base_key=base_key,
        x_query=x_query,
        rollout_times=cfg.rollout_times,
        rollout_length=cfg.rollout_length,
        progress_desc="Baseline rollouts",
    )
    tuned_beliefs, tuned_class_labels = collect_rollout_beliefs_with_factory(
        pred_rule_factory=tuned_factory,
        x_train=dgp.train_data["x"],
        y_train=dgp.train_data["y"],
        base_key=base_key,
        x_query=x_query,
        rollout_times=cfg.rollout_times,
        rollout_length=cfg.rollout_length,
        progress_desc="Tuned rollouts",
    )

    if not np.array_equal(baseline_class_labels, tuned_class_labels):
        raise ValueError(
            f"Class label mismatch: baseline={baseline_class_labels}, tuned={tuned_class_labels}"
        )

    _write_outputs(
        outdir=outdir,
        model_name="baseline",
        beliefs=baseline_beliefs,
        query_idx=query_idx,
        x_query=x_query,
        y_query=y_query,
        class_labels=baseline_class_labels,
        cfg=cfg,
    )
    _write_outputs(
        outdir=outdir,
        model_name="tuned",
        beliefs=tuned_beliefs,
        query_idx=query_idx,
        x_query=x_query,
        y_query=y_query,
        class_labels=tuned_class_labels,
        cfg=cfg,
    )
