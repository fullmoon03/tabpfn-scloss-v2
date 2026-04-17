from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .base import SyntheticGenerator, key_to_seed


@dataclass
class SCMDagNodeSpec:
    parents: list[int]
    noise_scale: float
    root_mode: str | None
    weights: np.ndarray | None
    nonlinearity: str | None
    interaction_pair: tuple[int, int] | None
    interaction_coef: float | None
    skip_parent_local_idx: int | None
    skip_coef: float | None


@dataclass
class SCMDagClassSpec:
    feature_idx: np.ndarray
    weights: np.ndarray
    nonlinearity: str
    interaction_pair: tuple[int, int] | None
    interaction_coef: float | None


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(float(temperature), 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _apply_random_nonlinearity(v: np.ndarray, kind: str) -> np.ndarray:
    if kind == "affine":
        return v
    if kind == "tanh":
        return np.tanh(v)
    if kind == "sin":
        return np.sin(v)
    if kind == "square":
        return np.sign(v) * (np.abs(v) ** 2)
    raise ValueError(f"Unknown nonlinearity: {kind}")


def _sample_labels_from_probs(
    probs: np.ndarray,
    rng: np.random.Generator,
    class_prior: np.ndarray,
    label_noise: float,
    deterministic_label_prob: float,
) -> np.ndarray:
    probs = probs.copy()
    probs = probs * class_prior[None, :]
    probs = probs / probs.sum(axis=1, keepdims=True)

    n, n_classes = probs.shape
    y = np.argmax(probs, axis=1).astype(np.int64)

    if deterministic_label_prob < 1.0:
        sample_mask = rng.random(n) > deterministic_label_prob
        if np.any(sample_mask):
            for i in np.where(sample_mask)[0]:
                y[i] = rng.choice(n_classes, p=probs[i])

    if label_noise > 0.0:
        flip_mask = rng.random(n) < label_noise
        if np.any(flip_mask):
            y[flip_mask] = rng.integers(0, n_classes, size=int(flip_mask.sum()))
    return y


def _random_dirichlet_prior(
    n_classes: int,
    rng: np.random.Generator,
    alpha_choices: tuple[float, ...],
) -> np.ndarray:
    alpha = float(rng.choice(alpha_choices))
    return rng.dirichlet(alpha * np.ones(n_classes))


class SCMDagClassificationGenerator(SyntheticGenerator):
    num_classes: int
    max_parents: int
    node_specs: list[SCMDagNodeSpec]
    class_specs: list[SCMDagClassSpec]
    temperature: float
    label_noise: float
    deterministic_label_prob: float
    class_prior: np.ndarray
    logit_scale: float
    logit_noise_std: float

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        seed: int | None = None,
        max_parents: int = 2,
    ):
        super().__init__(num_features)
        self.num_classes = num_classes
        self.max_parents = max_parents

        rng = np.random.default_rng(seed)
        self.temperature = float(rng.uniform(0.42, 1.0))
        self.label_noise = float(rng.uniform(0.0, 0.05))
        self.deterministic_label_prob = 0.65
        self.class_prior = _random_dirichlet_prior(
            num_classes, rng, alpha_choices=(6.0, 10.0, 14.0)
        ).astype(np.float64)
        self.logit_scale = float(rng.uniform(1.6, 3.0))
        self.logit_noise_std = 0.08

        self.node_specs = self._sample_node_specs(rng)
        self.class_specs = self._sample_class_specs(rng)
        self.metadata = {
            "generator": "scm_dag",
            "num_classes": self.num_classes,
            "max_parents": self.max_parents,
            "temperature": self.temperature,
            "label_noise": self.label_noise,
            "deterministic_label_prob": self.deterministic_label_prob,
            "class_prior": self.class_prior.tolist(),
            "logit_scale": self.logit_scale,
            "node_specs": [self._node_spec_to_metadata(spec) for spec in self.node_specs],
            "class_specs": [self._class_spec_to_metadata(spec) for spec in self.class_specs],
        }

    def _sample_node_specs(self, rng: np.random.Generator) -> list[SCMDagNodeSpec]:
        specs: list[SCMDagNodeSpec] = []
        for j in range(self.num_features):
            parents: list[int]
            if j == 0:
                parents = []
            else:
                k = int(rng.integers(0, min(self.max_parents, j) + 1))
                parents = [] if k == 0 else rng.choice(j, size=k, replace=False).tolist()

            noise_scale = float(rng.uniform(0.03, 0.14))
            if not parents:
                specs.append(
                    SCMDagNodeSpec(
                        parents=[],
                        noise_scale=noise_scale,
                        root_mode=str(rng.choice(["gauss", "tanhgauss"])),
                        weights=None,
                        nonlinearity=None,
                        interaction_pair=None,
                        interaction_coef=None,
                        skip_parent_local_idx=None,
                        skip_coef=None,
                    )
                )
                continue

            weights = rng.normal(0.0, 1.0, size=len(parents)).astype(np.float64)
            interaction_pair = None
            interaction_coef = None
            if len(parents) >= 2 and rng.random() < 0.2:
                pair = rng.choice(len(parents), size=2, replace=False)
                interaction_pair = (int(pair[0]), int(pair[1]))
                interaction_coef = float(rng.uniform(0.15, 0.55))

            skip_parent_local_idx = None
            skip_coef = None
            if rng.random() < 0.25:
                skip_parent_local_idx = int(rng.integers(0, len(parents)))
                skip_coef = float(rng.uniform(0.15, 0.55))

            specs.append(
                SCMDagNodeSpec(
                    parents=parents,
                    noise_scale=noise_scale,
                    root_mode=None,
                    weights=weights,
                    nonlinearity=str(rng.choice(["affine", "tanh"])),
                    interaction_pair=interaction_pair,
                    interaction_coef=interaction_coef,
                    skip_parent_local_idx=skip_parent_local_idx,
                    skip_coef=skip_coef,
                )
            )
        return specs

    def _sample_class_specs(self, rng: np.random.Generator) -> list[SCMDagClassSpec]:
        specs: list[SCMDagClassSpec] = []
        for _ in range(self.num_classes):
            k_min = max(2, min(3, self.num_features))
            k_max = min(self.num_features, 6)
            k = int(rng.integers(k_min, k_max + 1))
            idx = rng.choice(self.num_features, size=k, replace=False)
            weights = rng.normal(0.0, 1.0, size=k).astype(np.float64)
            interaction_pair = None
            interaction_coef = None
            if k >= 2 and rng.random() < 0.25:
                pair = rng.choice(k, size=2, replace=False)
                interaction_pair = (int(pair[0]), int(pair[1]))
                interaction_coef = float(rng.uniform(0.15, 0.45))
            specs.append(
                SCMDagClassSpec(
                    feature_idx=np.asarray(idx, dtype=np.int32),
                    weights=weights,
                    nonlinearity=str(rng.choice(["affine", "tanh"])),
                    interaction_pair=interaction_pair,
                    interaction_coef=interaction_coef,
                )
            )
        return specs

    def _generate_features(self, rng: np.random.Generator, n: int) -> np.ndarray:
        x = np.zeros((n, self.num_features), dtype=np.float64)
        for j, spec in enumerate(self.node_specs):
            if not spec.parents:
                if spec.root_mode == "gauss":
                    col = rng.normal(0.0, 1.0, size=n)
                else:
                    z = rng.normal(0.0, 1.0, size=n)
                    col = np.tanh(z) + 0.1 * z
                x[:, j] = col + rng.normal(0.0, spec.noise_scale, size=n)
                continue

            xp = x[:, spec.parents]
            base = xp @ np.asarray(spec.weights, dtype=np.float64)
            if spec.interaction_pair is not None and spec.interaction_coef is not None:
                i1, i2 = spec.interaction_pair
                base = base + spec.interaction_coef * xp[:, i1] * xp[:, i2]
            out = _apply_random_nonlinearity(base, str(spec.nonlinearity))
            if spec.skip_parent_local_idx is not None and spec.skip_coef is not None:
                out = out + spec.skip_coef * xp[:, spec.skip_parent_local_idx]
            x[:, j] = out + rng.normal(0.0, spec.noise_scale, size=n)

        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
        return x.astype(np.float32)

    def sample_x(self, key: PRNGKeyArray, n: int) -> Array:
        seed = key_to_seed(key)
        rng = np.random.default_rng(seed)
        return self._generate_features(rng, n)

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        seed = key_to_seed(key)
        rng = np.random.default_rng(seed)
        x = self._generate_features(rng, n).astype(np.float64)

        logits = np.zeros((n, self.num_classes), dtype=np.float64)
        for class_idx, spec in enumerate(self.class_specs):
            idx = np.asarray(spec.feature_idx, dtype=np.int32)
            z = x[:, idx] @ np.asarray(spec.weights, dtype=np.float64)
            z = _apply_random_nonlinearity(z, spec.nonlinearity)
            if spec.interaction_pair is not None and spec.interaction_coef is not None:
                i1, i2 = spec.interaction_pair
                z = z + spec.interaction_coef * x[:, idx[i1]] * x[:, idx[i2]]
            logits[:, class_idx] = self.logit_scale * z + rng.normal(
                0.0, self.logit_noise_std, size=n
            )

        probs = _softmax(logits, temperature=self.temperature)
        y = _sample_labels_from_probs(
            probs,
            rng,
            class_prior=self.class_prior,
            label_noise=self.label_noise,
            deterministic_label_prob=self.deterministic_label_prob,
        )

        metadata = {
            **self.metadata,
            "num_samples": int(n),
            "class_counts": {
                int(cls): int((y == cls).sum()) for cls in np.unique(y)
            },
        }
        return {
            "x": x,
            "y": y.astype(np.int16),
            "metadata": metadata,
        }

    @staticmethod
    def _node_spec_to_metadata(spec: SCMDagNodeSpec) -> dict[str, Any]:
        return {
            "parents": list(spec.parents),
            "noise_scale": float(spec.noise_scale),
            "root_mode": spec.root_mode,
            "weights": None if spec.weights is None else np.asarray(spec.weights).tolist(),
            "nonlinearity": spec.nonlinearity,
            "interaction_pair": spec.interaction_pair,
            "interaction_coef": spec.interaction_coef,
            "skip_parent_local_idx": spec.skip_parent_local_idx,
            "skip_coef": spec.skip_coef,
        }

    @staticmethod
    def _class_spec_to_metadata(spec: SCMDagClassSpec) -> dict[str, Any]:
        return {
            "feature_idx": np.asarray(spec.feature_idx).tolist(),
            "weights": np.asarray(spec.weights).tolist(),
            "nonlinearity": spec.nonlinearity,
            "interaction_pair": spec.interaction_pair,
            "interaction_coef": spec.interaction_coef,
        }
