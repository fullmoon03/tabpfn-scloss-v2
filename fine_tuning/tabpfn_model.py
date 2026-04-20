from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from torch import Tensor, nn

from fine_tuning.preprocess import PreprocessedPrefix, transform_query


class TrainableTabPFNClassifier(nn.Module):
    """Trainable wrapper around TabPFNClassifier."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        n_estimators: int,
        average_before_softmax: bool,
        categorical_features_indices: list[int],
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.estimator = TabPFNClassifier(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=1.0,
            categorical_features_indices=categorical_features_indices,
            fit_mode="batched",
            model_path=checkpoint_path,
            device="auto" if device is None else device,
            differentiable_input=False,
        )
        self.estimator._initialize_model_variables()
        self.estimator.softmax_temperature_ = self.estimator.softmax_temperature
        self.models = nn.ModuleList(self.estimator.models_)

    def query_probabilities(
        self,
        *,
        preprocessed_prefix: PreprocessedPrefix,
        x_query: np.ndarray,
        n_classes: int,
    ) -> Tensor:
        device = next(self.parameters()).device
        x_context = [x.to(device) for x in preprocessed_prefix.x_context]
        y_context = [y.to(device) for y in preprocessed_prefix.y_context]
        x_query_preprocessed = [
            x.to(device) for x in transform_query(preprocessed_prefix, x_query)
        ]

        self.estimator.classes_ = preprocessed_prefix.classes
        self.estimator.n_classes_ = len(preprocessed_prefix.classes)
        self.estimator.class_counts_ = preprocessed_prefix.class_counts
        self.estimator.models_ = list(self.models)
        self.estimator.fit_from_preprocessed(
            x_context,
            y_context,
            preprocessed_prefix.cat_ix,
            preprocessed_prefix.configs,
        )
        probs_bcq = self.estimator.forward(
            x_query_preprocessed,
            use_inference_mode=False,
            return_logits=False,
        )
        probs_qc_local = probs_bcq.squeeze(0).transpose(0, 1)

        probs = probs_qc_local.new_zeros((probs_qc_local.shape[0], n_classes))
        class_indices = torch.as_tensor(
            preprocessed_prefix.classes.astype(np.int64),
            dtype=torch.long,
            device=probs_qc_local.device,
        )
        probs[:, class_indices] = probs_qc_local
        return probs
