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
