from __future__ import annotations

import warnings
from typing import Callable

import jax
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from dgp import OPENML_BINARY_CLASSIFICATION, OPENML_CLASSIFICATION
from rollout import TabPFNClassifierPredRule, get_x_new

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def make_classifier_pred_rule(cfg: DictConfig, dgp: object) -> TabPFNClassifierPredRule:
    dim_x = dgp.train_data["x"].shape[-1]
    if cfg.dgp.name.startswith("classification-fixed") or cfg.dgp.name == "classification-scm":
        categorical_x = [False] * dim_x
    elif cfg.dgp.name in OPENML_CLASSIFICATION + OPENML_BINARY_CLASSIFICATION:
        categorical_x = dgp.categorical_x
    else:
        raise ValueError(
            "Fixed-query rollout currently supports classification datasets only, "
            f"got {cfg.dgp.name}."
        )
    return TabPFNClassifierPredRule(
        categorical_x, cfg.n_estimators, cfg.average_before_softmax
    )


def sample_test_queries(
    test_data: dict[str, np.ndarray], num_queries: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_test = test_data["x"].shape[0]
    if n_test < num_queries:
        raise ValueError(f"Need at least {num_queries} test queries, found {n_test}.")

    rng = np.random.default_rng(seed)
    query_idx = np.sort(rng.choice(n_test, size=num_queries, replace=False))
    return query_idx, test_data["x"][query_idx], test_data["y"][query_idx]


def query_belief(
    pred_rule: TabPFNClassifierPredRule,
    x_query: np.ndarray,
) -> np.ndarray:
    return pred_rule.predict_proba(x_query)


def single_rollout_belief_trajectory(
    key: jax.Array,
    pred_rule: TabPFNClassifierPredRule,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    rollout_length: int,
) -> np.ndarray:
    x_context = np.array(x_train, copy=True)
    y_context = np.array(y_train, copy=True)

    pred_rule.fit(x_context, y_context)
    init_belief = query_belief(pred_rule, x_query)
    beliefs = np.empty(
        (rollout_length + 1, x_query.shape[0], init_belief.shape[1]), dtype=np.float64
    )
    beliefs[0] = init_belief

    for depth in range(1, rollout_length + 1):
        rkey = jax.random.fold_in(key, depth)
        rkey, subkey = jax.random.split(rkey)
        x_new = np.asarray(get_x_new(subkey, x_context))

        rkey, subkey = jax.random.split(rkey)
        y_new = pred_rule.sample(subkey, x_new, x_context, y_context)

        x_context = np.concatenate([x_context, x_new], axis=0)
        y_context = np.concatenate([y_context, np.atleast_1d(y_new)], axis=0)

        pred_rule.fit(x_context, y_context)
        beliefs[depth] = query_belief(pred_rule, x_query)

    return beliefs


def collect_rollout_beliefs(
    cfg: DictConfig,
    dgp: object,
    base_key: jax.Array,
    x_query: np.ndarray,
    progress_desc: str = "Rollouts",
) -> tuple[np.ndarray, np.ndarray]:
    pred_rule = make_classifier_pred_rule(cfg, dgp)
    pred_rule.fit(dgp.train_data["x"], dgp.train_data["y"])
    class_labels = np.asarray(pred_rule.classes_)

    beliefs = np.empty(
        (
            cfg.rollout_times,
            cfg.rollout_length + 1,
            x_query.shape[0],
            class_labels.shape[0],
        ),
        dtype=np.float64,
    )

    for rollout_idx in tqdm(range(cfg.rollout_times), desc=progress_desc):
        pred_rule = make_classifier_pred_rule(cfg, dgp)
        rollout_key = jax.random.fold_in(base_key, rollout_idx)
        beliefs[rollout_idx] = single_rollout_belief_trajectory(
            rollout_key,
            pred_rule,
            dgp.train_data["x"],
            dgp.train_data["y"],
            x_query,
            cfg.rollout_length,
        )

    return beliefs, class_labels
