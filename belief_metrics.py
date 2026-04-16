from __future__ import annotations

import numpy as np


def _validate_probability_trajectories(mean_belief: np.ndarray) -> None:
    if mean_belief.ndim != 3:
        raise ValueError(
            "mean_belief must have shape (num_depths, num_queries, num_classes)."
        )


def probability_distance(
    p: np.ndarray, q: np.ndarray, distance_name: str
) -> np.ndarray:
    try:
        np.broadcast_shapes(p.shape, q.shape)
    except ValueError as exc:
        raise ValueError(
            "Probability tensors must be broadcast-compatible, "
            f"got {p.shape} and {q.shape}."
        ) from exc
    diff = p - q
    if distance_name == "l1":
        return np.abs(diff).sum(axis=-1)
    if distance_name == "tv":
        return 0.5 * np.abs(diff).sum(axis=-1)
    if distance_name == "l2":
        return np.sqrt((diff * diff).sum(axis=-1))
    if distance_name == "linf":
        return np.abs(diff).max(axis=-1)

    raise ValueError(
        f"Unsupported probability distance '{distance_name}'. "
        "Supported distances: l1, tv, l2, linf."
    )


def compute_expected_martingale_drift(
    mean_belief: np.ndarray,
    distance_name: str = "l1",
    reference_depth: int = 0,
    average_from_depth: int = 1,
) -> dict[str, np.ndarray | float | int | str]:
    _validate_probability_trajectories(mean_belief)

    num_depths = mean_belief.shape[0]
    if not 0 <= reference_depth < num_depths:
        raise ValueError(f"reference_depth must be in [0, {num_depths}), got {reference_depth}.")
    if not 0 <= average_from_depth < num_depths:
        raise ValueError(
            f"average_from_depth must be in [0, {num_depths}), got {average_from_depth}."
        )

    reference_belief = mean_belief[reference_depth]
    full_distance_by_depth = probability_distance(
        mean_belief,
        reference_belief[None, :, :],
        distance_name=distance_name,
    )
    distance_depths = np.arange(reference_depth + 1, num_depths, dtype=np.int32)
    distance_by_depth = full_distance_by_depth[reference_depth + 1 :]

    averaging_start_depth = max(average_from_depth, reference_depth + 1)
    if averaging_start_depth >= num_depths:
        raise ValueError(
            "Need at least one depth after the averaging start. "
            f"Got reference_depth={reference_depth}, average_from_depth={average_from_depth}, "
            f"num_depths={num_depths}."
        )

    emd_per_query = full_distance_by_depth[averaging_start_depth:].mean(axis=0)
    global_emd = float(emd_per_query.mean())

    return {
        "metric_name": "expected_martingale_drift",
        "distance_name": distance_name,
        "reference_depth": reference_depth,
        "average_from_depth": average_from_depth,
        "averaging_start_depth": averaging_start_depth,
        "distance_depths": distance_depths,
        "distance_by_depth": distance_by_depth,
        "emd_per_query": emd_per_query,
        "global_emd": global_emd,
    }


def get_true_class_indices(
    class_labels: np.ndarray, y_query: np.ndarray
) -> np.ndarray:
    class_labels = np.asarray(class_labels)
    y_query = np.asarray(y_query)
    true_class_idx = np.full(y_query.shape[0], -1, dtype=np.int32)

    for query_idx, y_true in enumerate(y_query):
        matches = np.flatnonzero(class_labels == y_true)
        if matches.size != 1:
            raise ValueError(
                f"Expected exactly one matching class label for query {query_idx}, got {matches.size}."
            )
        true_class_idx[query_idx] = int(matches[0])

    return true_class_idx


def extract_true_label_probabilities(
    beliefs: np.ndarray,
    class_labels: np.ndarray,
    y_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if beliefs.ndim != 4:
        raise ValueError(
            "beliefs must have shape (num_rollouts, num_depths, num_queries, num_classes)."
        )

    true_class_idx = get_true_class_indices(class_labels, y_query)
    num_rollouts, num_depths, num_queries, _ = beliefs.shape
    theta_true = np.empty((num_rollouts, num_depths, num_queries), dtype=np.float64)

    for query_idx, class_idx in enumerate(true_class_idx):
        theta_true[:, :, query_idx] = beliefs[:, :, query_idx, class_idx]

    return theta_true, true_class_idx


def compute_true_label_variance_trajectory(
    beliefs: np.ndarray,
    class_labels: np.ndarray,
    y_query: np.ndarray,
    ddof: int = 1,
    start_depth: int = 1,
) -> dict[str, np.ndarray | float | int]:
    theta_true, true_class_idx = extract_true_label_probabilities(
        beliefs, class_labels, y_query
    )

    num_rollouts, num_depths, num_queries = theta_true.shape
    if start_depth < 0 or start_depth >= num_depths:
        raise ValueError(
            f"start_depth must be in [0, {num_depths}), got {start_depth}."
        )
    if num_rollouts <= ddof:
        raise ValueError(
            f"Need num_rollouts > ddof to compute sample variance, got {num_rollouts} and {ddof}."
        )

    depths = np.arange(start_depth, num_depths, dtype=np.int32)
    theta_by_depth = theta_true[:, start_depth:, :]
    variance_by_depth = np.var(theta_by_depth, axis=0, ddof=ddof)
    mean_by_depth = np.mean(theta_by_depth, axis=0)
    variance_by_query = variance_by_depth.T
    mean_by_query = mean_by_depth.T
    peak_step = depths[np.argmax(variance_by_query, axis=1)]
    max_variance = np.max(variance_by_query, axis=1)

    return {
        "metric_name": "true_label_probability_variance",
        "ddof": ddof,
        "start_depth": start_depth,
        "depths": depths,
        "true_class_idx": true_class_idx,
        "theta_true": theta_true,
        "theta_by_depth": theta_by_depth,
        "variance_by_depth": variance_by_depth,
        "variance_by_query": variance_by_query,
        "mean_by_depth": mean_by_depth,
        "mean_by_query": mean_by_query,
        "peak_step": peak_step,
        "max_variance": max_variance,
    }
