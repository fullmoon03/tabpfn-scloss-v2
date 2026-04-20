from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch

from fine_tuning.tabpfn_model import TrainableTabPFNClassifier
from fine_tuning.preprocess import TabPFNPreprocessor


class FineTunedTabPFNClassifierPredRule:
    """`fit/predict_proba/sample` adapter for a martingale-finetuned TabPFN."""

    def __init__(
        self,
        *,
        base_checkpoint_path: str | Path,
        tuned_checkpoint_path: str | Path,
        class_labels: np.ndarray,
        categorical_features_indices: list[int] | None = None,
        n_estimators: int = 4,
        average_before_softmax: bool = False,
        device: str | torch.device | None = None,
        torch_seed: int = 0,
    ) -> None:
        self.classes_ = np.asarray(class_labels)
        self._class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        self._device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._torch_seed = torch_seed
        self._base_checkpoint_path = base_checkpoint_path
        self._categorical_features_indices = categorical_features_indices or []
        self._n_estimators = n_estimators
        self._average_before_softmax = average_before_softmax
        self._model = TrainableTabPFNClassifier(
            checkpoint_path=base_checkpoint_path,
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            categorical_features_indices=self._categorical_features_indices,
            device=self._device,
        )
        payload = torch.load(
            tuned_checkpoint_path,
            map_location=self._device,
            weights_only=False,
        )
        state_dict = (
            payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        self._preprocessor = TabPFNPreprocessor(
            categorical_features_indices=self._categorical_features_indices,
            n_estimators=self._n_estimators,
            average_before_softmax=self._average_before_softmax,
            model_path=self._base_checkpoint_path,
            device=self._device,
        )
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FineTunedTabPFNClassifierPredRule":
        self._x_train = np.asarray(x, dtype=np.float32)
        self._y_train = np.asarray(y)
        return self

    def _encoded_y_train(self) -> np.ndarray:
        if self._y_train is None:
            raise RuntimeError("Call fit before predict_proba.")
        return np.asarray(
            [self._class_to_index[label] for label in self._y_train],
            dtype=np.int64,
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._x_train is None:
            raise RuntimeError("Call fit before predict_proba.")

        torch.manual_seed(self._torch_seed)
        preprocessed_prefix = self._preprocessor.fit_prefix(
            self._x_train,
            self._encoded_y_train(),
        )

        with torch.inference_mode():
            probs = self._model.query_probabilities(
                preprocessed_prefix=preprocessed_prefix,
                x_query=np.asarray(x, dtype=np.float32),
                n_classes=len(self.classes_),
            )
        return probs.detach().cpu().numpy()

    def sample(
        self,
        key: jax.Array,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> Any:
        self.fit(x_prev, y_prev)
        probs_new = self.predict_proba(x_new).squeeze()
        idx_new = jax.random.choice(key, a=self.classes_.size, p=probs_new)
        y_new = self.classes_[idx_new]
        return y_new.squeeze() if isinstance(y_new, np.ndarray) else y_new
