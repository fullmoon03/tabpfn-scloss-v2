from __future__ import annotations

import csv
import os

import numpy as np


def write_query_metric_csv(
    path: str,
    value_prefix: str,
    query_idx: np.ndarray,
    y_query: np.ndarray,
    values_by_query: np.ndarray,
    start_index: int = 0,
) -> None:
    fieldnames = ["query_id", "y_true"] + [
        f"{value_prefix}{idx}"
        for idx in range(start_index, start_index + values_by_query.shape[1])
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
            for value_idx, value in enumerate(
                values_by_query[row_idx], start=start_index
            ):
                row[f"{value_prefix}{value_idx}"] = float(value)
            writer.writerow(row)


def write_query_summary_csv(
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
