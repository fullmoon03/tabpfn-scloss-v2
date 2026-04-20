from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.preprocessing.datamodel import FeatureModality
from torch import Tensor


@dataclass
class PreprocessedPrefix:
    members: list[Any]
    classes: np.ndarray
    class_counts: np.ndarray
    x_context: list[Tensor]
    y_context: list[Tensor]
    cat_ix: list[list[list[int]]]
    configs: list[list[Any]]


class TabPFNPreprocessor:
    """Fit TabPFN preprocessing for one context/prefix."""

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

    def fit_prefix(
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
    ) -> PreprocessedPrefix:
        """Fit TabPFN preprocessing for a context."""

        x_context = np.asarray(x_context)
        y_context = np.asarray(y_context)

        self._clf.fit(x_context, y_context)
        executor = self._clf.executor_
        if not hasattr(executor, "ensemble_members"):
            raise RuntimeError(
                "Expected TabPFNClassifier(fit_mode='fit_preprocessors') to expose "
                "executor_.ensemble_members."
            )
        return PreprocessedPrefix(
            members=list(executor.ensemble_members),
            classes=np.asarray(self._clf.classes_),
            class_counts=np.asarray(self._clf.class_counts_),
            x_context=[
                _to_tensor(
                    member.X_train,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                ).unsqueeze(0)
                for member in executor.ensemble_members
            ],
            y_context=[
                _to_tensor(
                    member.y_train,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                ).unsqueeze(0)
                for member in executor.ensemble_members
            ],
            cat_ix=[
                [
                    member.feature_schema.indices_for(FeatureModality.CATEGORICAL)
                    for member in executor.ensemble_members
                ]
            ],
            configs=[[member.config] for member in executor.ensemble_members],
        )


def make_preprocessor_factory(
    *,
    categorical_features_indices: list[int],
    n_estimators: int,
    average_before_softmax: bool,
    model_path: str | Path,
    device: str | torch.device | None = None,
) -> Callable[[], TabPFNPreprocessor]:
    def factory() -> TabPFNPreprocessor:
        return TabPFNPreprocessor(
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


def transform_query(
    preprocessed_prefix: PreprocessedPrefix,
    x_query: np.ndarray,
) -> list[Tensor]:
    query_tensors: list[Tensor] = []
    for member in preprocessed_prefix.members:
        x_query_pre = member.transform_X_test(np.asarray(x_query))
        query_tensors.append(
            _to_tensor(
                x_query_pre,
                dtype=torch.float32,
                device=torch.device("cpu"),
            ).unsqueeze(0)
        )
    return query_tensors
