from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.inference import _maybe_run_gpu_preprocessing
from torch import Tensor


@dataclass
class CachedPrefix:
    members: list[Any]
    classes: np.ndarray


class StandardTabPFNPrefixCache:
    """Cache official TabPFN preprocessing for one context/prefix at a time."""

    def __init__(
        self,
        *,
        categorical_features_indices: list[int],
        n_estimators: int,
        average_before_softmax: bool,
        model_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self._clf = TabPFNClassifier(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=1.0,
            categorical_features_indices=categorical_features_indices,
            fit_mode="fit_preprocessors",
            model_path=model_path,
            device="cpu" if device is None else device,
        )
        self._cached: dict[tuple[int, bytes, bytes], CachedPrefix] = {}

    def fit_prefix(self, x_context: np.ndarray, y_context: np.ndarray) -> CachedPrefix:
        """Fit official TabPFN preprocessing for a context and cache the result."""

        x_context = np.asarray(x_context)
        y_context = np.asarray(y_context)
        cache_key = (
            x_context.shape[0],
            np.ascontiguousarray(x_context).view(np.uint8).tobytes(),
            np.ascontiguousarray(y_context).view(np.uint8).tobytes(),
        )
        cached = self._cached.get(cache_key)
        if cached is not None:
            return cached

        self._clf.fit(x_context, y_context)
        executor = self._clf.executor_
        if not hasattr(executor, "ensemble_members"):
            raise RuntimeError(
                "Expected TabPFNClassifier(fit_mode='fit_preprocessors') to cache "
                "executor_.ensemble_members."
            )
        cached = CachedPrefix(
            members=list(executor.ensemble_members),
            classes=np.asarray(self._clf.classes_),
        )
        self._cached[cache_key] = cached
        return cached


def make_standard_prefix_cache_factory(
    *,
    categorical_features_indices: list[int],
    n_estimators: int,
    average_before_softmax: bool,
    model_path: str | Path,
    device: str | torch.device | None = None,
):
    def factory() -> StandardTabPFNPrefixCache:
        return StandardTabPFNPrefixCache(
            categorical_features_indices=categorical_features_indices,
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            model_path=model_path,
            device=device,
        )

    return factory


def _to_tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _local_logits_from_global(
    logits: Tensor,
    *,
    class_permutation: np.ndarray | None,
    n_local_classes: int,
) -> Tensor:
    if class_permutation is None:
        return logits[..., :n_local_classes]

    if len(class_permutation) != n_local_classes:
        use_perm = np.arange(n_local_classes)
        use_perm[: len(class_permutation)] = class_permutation
    else:
        use_perm = class_permutation
    return logits[..., torch.as_tensor(use_perm, dtype=torch.long, device=logits.device)]


def query_probabilities_from_cached_prefix(
    model: torch.nn.Module,
    *,
    cached_prefix: CachedPrefix,
    x_query: np.ndarray,
    n_classes: int,
    average_before_softmax: bool,
) -> Tensor:
    """Run the trainable model on official TabPFN-preprocessed prefix/query data."""

    device = next(model.parameters()).device
    estimator_logits: list[Tensor] = []
    n_local_classes = len(cached_prefix.classes)

    for member in cached_prefix.members:
        x_train = _to_tensor(member.X_train, dtype=torch.float32, device=device)
        x_query_pre = member.transform_X_test(np.asarray(x_query))
        x_query_tensor = _to_tensor(x_query_pre, dtype=torch.float32, device=device)
        y_train = _to_tensor(member.y_train, dtype=torch.float32, device=device)

        x_full = torch.cat([x_train, x_query_tensor], dim=0).unsqueeze(1)
        x_full = _maybe_run_gpu_preprocessing(
            x_full,
            gpu_preprocessor=member.gpu_preprocessor,
            num_train_rows=x_train.shape[0],
        ).transpose(0, 1)
        logits = model(x_num=x_full, y_train=y_train[None, :]).squeeze(0)
        estimator_logits.append(
            _local_logits_from_global(
                logits,
                class_permutation=member.config.class_permutation,
                n_local_classes=n_local_classes,
            )
        )

    stacked_logits = torch.stack(estimator_logits, dim=0)
    if average_before_softmax:
        local_probs = torch.softmax(stacked_logits.mean(dim=0), dim=-1)
    else:
        local_probs = torch.softmax(stacked_logits, dim=-1).mean(dim=0)

    probs = local_probs.new_zeros((local_probs.shape[0], n_classes))
    class_indices = torch.as_tensor(
        cached_prefix.classes.astype(np.int64),
        dtype=torch.long,
        device=local_probs.device,
    )
    probs[:, class_indices] = local_probs
    return probs
