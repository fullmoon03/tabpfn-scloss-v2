import csv
import logging
import os
import warnings

import hydra
import jax
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import belief_metrics
from fixed_query_rollout import make_classifier_pred_rule, sample_test_queries
import utils
from dgp import load_dgp
from rollout import forward_sampling, get_x_new

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def sample_reference_trajectory(
    cfg: DictConfig,
    dgp: object,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    pred_rule = make_classifier_pred_rule(cfg, dgp)
    return forward_sampling(
        key,
        pred_rule.sample,
        dgp.train_data["x"],
        dgp.train_data["y"],
        cfg.rollout_length,
        show_progress=False,
    )


def one_step_conditional_belief(
    cfg: DictConfig,
    dgp: object,
    key: jax.Array,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    pred_rule = make_classifier_pred_rule(cfg, dgp)
    rkey, x_key = jax.random.split(key)
    x_new = np.asarray(get_x_new(x_key, x_context))
    rkey, y_key = jax.random.split(rkey)
    y_new = pred_rule.sample(y_key, x_new, x_context, y_context)

    x_next = np.concatenate([x_context, x_new], axis=0)
    y_next = np.concatenate([y_context, np.atleast_1d(y_new)], axis=0)

    pred_rule = make_classifier_pred_rule(cfg, dgp)
    pred_rule.fit(x_next, y_next)
    return pred_rule.predict_proba(x_query)


def collect_conditional_beliefs(
    cfg: DictConfig,
    dgp: object,
    base_key: jax.Array,
    x_reference: np.ndarray,
    y_reference: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    n_train = dgp.train_data["x"].shape[0]
    num_depths = cfg.rollout_length + 1

    pred_rule = make_classifier_pred_rule(cfg, dgp)
    pred_rule.fit(dgp.train_data["x"], dgp.train_data["y"])
    num_classes = np.asarray(pred_rule.classes_).shape[0]

    conditional_beliefs = np.empty(
        (num_depths, cfg.num_posterior_samples, cfg.num_queries, num_classes),
        dtype=np.float64,
    )

    for depth in tqdm(range(num_depths), desc="Reference depths"):
        x_context = x_reference[: n_train + depth]
        y_context = y_reference[: n_train + depth]
        for sample_idx in range(cfg.num_posterior_samples):
            one_step_key = jax.random.fold_in(base_key, depth)
            one_step_key = jax.random.fold_in(one_step_key, sample_idx)
            conditional_beliefs[depth, sample_idx] = one_step_conditional_belief(
                cfg,
                dgp,
                one_step_key,
                x_context,
                y_context,
                x_query,
            )

    return conditional_beliefs


def write_variance_csv(
    path: str,
    prefix: str,
    query_idx: np.ndarray,
    y_query: np.ndarray,
    values_by_query: np.ndarray,
) -> None:
    fieldnames = ["query_id", "y_true"] + [
        f"{prefix}{depth}" for depth in range(values_by_query.shape[1])
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(values_by_query.shape[0]):
            row = {
                "query_id": int(query_idx[row_idx]),
                "y_true": y_query[row_idx],
            }
            for depth_idx, value in enumerate(values_by_query[row_idx]):
                row[f"{prefix}{depth_idx}"] = float(value)
            writer.writerow(row)


def write_summary_csv(
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


@hydra.main(version_base=None, config_path="conf", config_name="conditional-theta-variance")
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
    base_key, data_key, reference_key, continuation_key = jax.random.split(base_key, 4)

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

    x_reference, y_reference = sample_reference_trajectory(cfg, dgp, reference_key)
    utils.write_to(
        f"{outdir}/reference-trajectory.pickle",
        {"x": x_reference, "y": y_reference},
        verbose=True,
    )

    conditional_beliefs = collect_conditional_beliefs(
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

    theta_samples, true_class_idx = belief_metrics.select_true_label_probabilities(
        conditional_beliefs,
        class_labels,
        y_query,
    )
    theta_by_depth = theta_samples.mean(axis=1)
    variance_summary = belief_metrics.compute_query_variance_summary(
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

    write_variance_csv(
        f"{outdir}/conditional-theta-variance.csv",
        "var_t",
        query_idx,
        y_query,
        variance_summary["variance_by_query"],
    )
    write_summary_csv(
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


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
