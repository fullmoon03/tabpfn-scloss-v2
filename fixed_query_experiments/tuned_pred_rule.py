from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch

from fine_tuning import TabPFN2, TabPFN2Config


class FineTunedTabPFNClassifierPredRule:
    """`fit/predict_proba/sample` adapter for a martingale-finetuned TabPFN2."""

    def __init__(
        self,
        *,
        base_checkpoint_path: str | Path,
        tuned_checkpoint_path: str | Path,
        n_num_features: int,
        class_labels: np.ndarray,
        device: str | torch.device | None = None,
        torch_seed: int = 0,
    ) -> None:
        self.classes_ = np.asarray(class_labels)
        self._class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._torch_seed = torch_seed
        self._model = TabPFN2(
            TabPFN2Config(
                checkpoint_path=base_checkpoint_path,
                is_regression=False,
                n_num_features=n_num_features,
                n_classes=len(self.classes_),
            )
        )
        payload = torch.load(tuned_checkpoint_path, map_location=self._device, weights_only=False)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FineTunedTabPFNClassifierPredRule":
        self._x_train = np.asarray(x, dtype=np.float32)
        self._y_train = np.asarray(y)
        return self

    def _encoded_y_train(self) -> np.ndarray:
        if self._y_train is None:
            raise RuntimeError("Call fit before predict_proba.")
        return np.asarray([self._class_to_index[label] for label in self._y_train], dtype=np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._x_train is None:
            raise RuntimeError("Call fit before predict_proba.")

        torch.manual_seed(self._torch_seed)
        x_train = torch.as_tensor(self._x_train, dtype=torch.float32, device=self._device)
        y_train = torch.as_tensor(self._encoded_y_train(), dtype=torch.long, device=self._device)
        x_query = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=self._device)
        x_num = torch.cat([x_train[None, :, :], x_query[None, :, :]], dim=1)

        with torch.inference_mode():
            logits = self._model(x_num=x_num, y_train=y_train[None, :])[..., : len(self.classes_)]
            probs = logits.softmax(dim=-1).squeeze(0)
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
