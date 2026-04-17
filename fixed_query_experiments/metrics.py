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
        raise ValueError(
            f"reference_depth must be in [0, {num_depths}), got {reference_depth}."
        )
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


def select_true_label_probabilities(
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    y_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim < 2:
        raise ValueError(
            "probabilities must have shape (..., num_queries, num_classes)."
        )

    true_class_idx = get_true_class_indices(class_labels, y_query)
    true_label_probs = np.empty(probabilities.shape[:-1], dtype=np.float64)

    for query_idx, class_idx in enumerate(true_class_idx):
        true_label_probs[..., query_idx] = probabilities[..., query_idx, class_idx]

    return true_label_probs, true_class_idx


def extract_true_label_probabilities(
    beliefs: np.ndarray,
    class_labels: np.ndarray,
    y_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if beliefs.ndim != 4:
        raise ValueError(
            "beliefs must have shape (num_rollouts, num_depths, num_queries, num_classes)."
        )

    theta_true, true_class_idx = select_true_label_probabilities(
        beliefs, class_labels, y_query
    )
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

    num_rollouts, num_depths, _ = theta_true.shape
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


def compute_query_variance_summary(
    samples_by_depth: np.ndarray,
    depths: np.ndarray,
    ddof: int = 1,
) -> dict[str, np.ndarray | int]:
    if samples_by_depth.ndim != 3:
        raise ValueError(
            "samples_by_depth must have shape (num_depths, num_samples, num_queries)."
        )

    num_depths, num_samples, _ = samples_by_depth.shape
    depths = np.asarray(depths)
    if depths.shape != (num_depths,):
        raise ValueError(
            f"depths must have shape ({num_depths},), got {depths.shape}."
        )
    if num_samples <= ddof:
        raise ValueError(
            f"Need num_samples > ddof to compute sample variance, got {num_samples} and {ddof}."
        )

    variance_by_depth = np.var(samples_by_depth, axis=1, ddof=ddof)
    mean_by_depth = np.mean(samples_by_depth, axis=1)
    variance_by_query = variance_by_depth.T
    mean_by_query = mean_by_depth.T
    peak_step = depths[np.argmax(variance_by_query, axis=1)]
    max_variance = np.max(variance_by_query, axis=1)

    return {
        "ddof": ddof,
        "depths": depths,
        "variance_by_depth": variance_by_depth,
        "variance_by_query": variance_by_query,
        "mean_by_depth": mean_by_depth,
        "mean_by_query": mean_by_query,
        "peak_step": peak_step,
        "max_variance": max_variance,
    }


def compute_multiclass_accuracy(
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    y_true: np.ndarray,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pred_idx = np.argmax(probabilities, axis=1)
    pred_labels = np.asarray(class_labels)[pred_idx]
    return float(np.mean(pred_labels == np.asarray(y_true)))


def compute_multiclass_nll(
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-12,
) -> float:
    true_probs, _ = select_true_label_probabilities(
        probabilities[None, :, :], class_labels, y_true
    )
    true_probs = np.clip(true_probs[0], eps, 1.0)
    return float(-np.mean(np.log(true_probs)))


def compute_multiclass_ece(
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must have shape (num_queries, num_classes), got {probabilities.shape}."
        )
    pred_idx = np.argmax(probabilities, axis=1)
    pred_labels = np.asarray(class_labels)[pred_idx]
    confidences = probabilities[np.arange(probabilities.shape[0]), pred_idx]
    correctness = (pred_labels == np.asarray(y_true)).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_idx in range(n_bins):
        left = bin_edges[bin_idx]
        right = bin_edges[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences > left) & (confidences <= right)
        if not np.any(in_bin):
            continue
        bin_accuracy = correctness[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += in_bin.mean() * abs(bin_accuracy - bin_confidence)
    return float(ece)


def compute_calibration_metrics(
    probabilities: np.ndarray,
    class_labels: np.ndarray,
    y_true: np.ndarray,
    ece_bins: int = 10,
) -> dict[str, float]:
    return {
        "accuracy": compute_multiclass_accuracy(probabilities, class_labels, y_true),
        "nll": compute_multiclass_nll(probabilities, class_labels, y_true),
        "ece": compute_multiclass_ece(
            probabilities, class_labels, y_true, n_bins=ece_bins
        ),
    }
