from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class EncodedClassificationData:
    x_num: Tensor
    y: Tensor
    class_labels: np.ndarray


@dataclass
class ValTestQuerySplit:
    val_query_idx: np.ndarray
    test_query_idx: np.ndarray


def encode_classification_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    class_labels: np.ndarray | None = None,
    device: torch.device | str | None = None,
) -> EncodedClassificationData:
    class_labels = np.asarray(np.unique(y) if class_labels is None else class_labels)
    class_to_index = {label: idx for idx, label in enumerate(class_labels)}
    encoded_y = np.asarray([class_to_index[label] for label in y], dtype=np.int64)
    return EncodedClassificationData(
        x_num=torch.as_tensor(x, dtype=torch.float32, device=device),
        y=torch.as_tensor(encoded_y, dtype=torch.long, device=device),
        class_labels=class_labels,
    )


def sample_disjoint_val_test_queries(
    pool_size: int,
    query_size: int,
    seed: int,
) -> ValTestQuerySplit:
    if 2 * query_size > pool_size:
        raise ValueError(
            f"Need 2 * query_size <= pool_size for disjoint val/test queries, "
            f"got query_size={query_size}, pool_size={pool_size}."
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(pool_size, size=2 * query_size, replace=False)
    return ValTestQuerySplit(
        val_query_idx=np.sort(selected[:query_size]),
        test_query_idx=np.sort(selected[query_size:]),
    )


def make_eval_indices(
    pool_size: int,
    query_idx: np.ndarray,
    *,
    include_query_in_context: bool = True,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    query_idx = np.asarray(query_idx, dtype=np.int64)
    if include_query_in_context:
        context_idx = np.arange(pool_size, dtype=np.int64)
    else:
        mask = np.ones(pool_size, dtype=bool)
        mask[query_idx] = False
        context_idx = np.flatnonzero(mask)
    return (
        torch.as_tensor(context_idx[None, :], dtype=torch.long, device=device),
        torch.as_tensor(query_idx[None, :], dtype=torch.long, device=device),
    )


def gather_rows(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim == 1:
        return values[indices]
    if values.ndim == 2:
        return values[indices]
    raise ValueError(f"Expected values with 1 or 2 dims, got shape {tuple(values.shape)}.")
