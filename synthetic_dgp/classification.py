from __future__ import annotations

import jax
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .base import SyntheticGenerator, key_to_seed


class _LinearClassificationGenerator(SyntheticGenerator):
    beta0: np.ndarray

    def __init__(self, dim_x: int, beta0: np.ndarray):
        super().__init__(dim_x)
        self.beta0 = np.asarray(beta0, dtype=np.float64)

    def sample_x(self, key: PRNGKeyArray, n: int) -> Array:
        key, subkey = jax.random.split(key)
        return jax.random.uniform(subkey, shape=(n, self.num_features), minval=-1, maxval=1)


class ClassificationFixedGenerator(_LinearClassificationGenerator):
    def __init__(self, dim_x: int):
        fixed_key = jax.random.key(1058)
        beta0 = jax.random.uniform(fixed_key, shape=(dim_x,), minval=-2, maxval=3)
        super().__init__(dim_x, np.asarray(beta0, dtype=np.float64))

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        probs = jax.scipy.special.expit(x @ self.beta0)
        y = jax.random.bernoulli(y_key, probs)
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.int8),
        }


class ClassificationFixedGMMLinkGenerator(_LinearClassificationGenerator):
    a: float

    def __init__(self, dim_x: int, a: float):
        fixed_key = jax.random.key(1058)
        beta0 = jax.random.uniform(fixed_key, shape=(dim_x,), minval=-2, maxval=3)
        super().__init__(dim_x, np.asarray(beta0, dtype=np.float64))
        self.a = a

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        cdf = jax.scipy.stats.norm.cdf
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        link = lambda p: 0.7 * cdf(p, loc=self.a) + 0.3 * cdf(p, loc=2.0)
        probs = link(x @ self.beta0)
        y = jax.random.bernoulli(y_key, probs)
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.int8),
        }


class ClassificationPriorGenerator(_LinearClassificationGenerator):
    def __init__(self, key: PRNGKeyArray, dim_x: int):
        beta0 = jax.random.normal(key, shape=(dim_x,))
        super().__init__(dim_x, np.asarray(beta0, dtype=np.float64))

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        probs = jax.scipy.special.expit(x @ self.beta0)
        y = jax.random.bernoulli(y_key, probs)
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.int8),
        }


def _make_random_spd_cov(d: int, rng: np.random.Generator, jitter: float = 1e-3) -> np.ndarray:
    a = rng.normal(size=(d, d))
    cov = a @ a.T
    cov = cov / np.trace(cov) * d
    cov = cov + jitter * np.eye(d)
    return cov


def _sample_multivariate_student_t(
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
    df: float,
    cov: np.ndarray,
) -> np.ndarray:
    z = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov, size=n_samples)
    g = rng.chisquare(df, size=(n_samples, 1))
    scale = np.sqrt(np.clip(g / df, 1e-12, None))
    return z / scale


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(float(temperature), 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _sample_labels_from_probs(
    probs: np.ndarray,
    rng: np.random.Generator,
    class_prior: np.ndarray,
    label_noise: float,
    deterministic_label_prob: float,
) -> np.ndarray:
    probs = probs * class_prior[None, :]
    probs = probs / probs.sum(axis=1, keepdims=True)

    n, n_classes = probs.shape
    y = np.argmax(probs, axis=1).astype(np.int64)
    if deterministic_label_prob < 1.0:
        sample_mask = rng.random(n) > deterministic_label_prob
        if np.any(sample_mask):
            for idx in np.where(sample_mask)[0]:
                y[idx] = rng.choice(n_classes, p=probs[idx])

    if label_noise > 0.0:
        flip_mask = rng.random(n) < label_noise
        if np.any(flip_mask):
            y[flip_mask] = rng.integers(0, n_classes, size=int(flip_mask.sum()))
    return y


class ClassificationLinearGenerator(SyntheticGenerator):
    num_classes: int
    informative_idx: np.ndarray
    class_weights: np.ndarray
    class_bias: np.ndarray
    interaction_terms: list[list[tuple[int, int, float]]]
    feature_mode: str
    feature_params: dict[str, object]
    class_prior: np.ndarray
    temperature: float
    label_noise: float
    deterministic_label_prob: float
    logit_scale: float

    def __init__(
        self,
        dim_x: int,
        num_classes: int,
        seed: int = 1058,
        informative_min: int = 4,
        informative_max: int = 7,
        interaction_prob: float = 0.12,
    ):
        super().__init__(dim_x)
        self.num_classes = num_classes
        rng = np.random.default_rng(seed)

        informative = int(
            rng.integers(min(informative_min, dim_x), min(informative_max, dim_x) + 1)
        )
        self.informative_idx = np.sort(
            rng.choice(dim_x, size=informative, replace=False).astype(np.int32)
        )
        self.class_weights = rng.normal(0.0, 1.0, size=(num_classes, informative)).astype(
            np.float64
        )
        self.class_bias = rng.normal(0.0, 0.15, size=(num_classes,)).astype(np.float64)

        self.interaction_terms = []
        for _ in range(num_classes):
            class_terms: list[tuple[int, int, float]] = []
            if informative >= 2:
                for i in range(informative):
                    for j in range(i + 1, informative):
                        if rng.random() < interaction_prob:
                            class_terms.append((i, j, float(rng.uniform(-0.35, 0.35))))
            self.interaction_terms.append(class_terms)

        self.feature_mode = str(
            rng.choice(
                ["correlated_gaussian", "independent_hetero", "heavy_tail"],
                p=[0.6, 0.2, 0.2],
            )
        )
        self.feature_params = self._sample_feature_params(rng)
        alpha = float(rng.choice((6.0, 10.0, 14.0)))
        self.class_prior = rng.dirichlet(alpha * np.ones(num_classes)).astype(np.float64)
        self.temperature = float(rng.uniform(0.42, 1.0))
        self.label_noise = float(rng.uniform(0.0, 0.05))
        self.deterministic_label_prob = 0.65
        self.logit_scale = float(rng.uniform(1.6, 3.0))
        self.metadata = {
            "generator": "classification-linear",
            "num_classes": num_classes,
            "informative_idx": self.informative_idx.tolist(),
            "feature_mode": self.feature_mode,
            "temperature": self.temperature,
            "label_noise": self.label_noise,
            "deterministic_label_prob": self.deterministic_label_prob,
            "class_prior": self.class_prior.tolist(),
            "logit_scale": self.logit_scale,
        }

    def _sample_feature_params(self, rng: np.random.Generator) -> dict[str, object]:
        if self.feature_mode == "correlated_gaussian":
            return {
                "cov": _make_random_spd_cov(self.num_features, rng),
                "mean": rng.normal(scale=0.5, size=self.num_features).astype(np.float64),
            }
        if self.feature_mode == "independent_hetero":
            params: dict[str, object] = {
                "scales": np.exp(rng.normal(loc=0.0, scale=0.5, size=self.num_features)).astype(
                    np.float64
                ),
                "means": rng.normal(scale=0.5, size=self.num_features).astype(np.float64),
            }
            if self.num_features >= 4 and rng.random() < 0.4:
                k = int(rng.integers(1, max(2, self.num_features // 4)))
                idx = np.sort(rng.choice(self.num_features, size=k, replace=False).astype(np.int32))
                params["near_constant_idx"] = idx
                params["near_constant_scale"] = float(rng.uniform(0.02, 0.15))
            return params
        if self.feature_mode == "heavy_tail":
            n_skew = (
                int(rng.integers(1, max(2, self.num_features // 2) + 1))
                if self.num_features >= 2
                else 0
            )
            return {
                "cov": _make_random_spd_cov(self.num_features, rng),
                "df": float(rng.uniform(3.0, 6.0)),
                "skew_idx": np.sort(
                    rng.choice(self.num_features, size=n_skew, replace=False).astype(np.int32)
                )
                if n_skew > 0
                else np.asarray([], dtype=np.int32),
                "skew_mode": str(rng.choice(["exp", "softplus"])),
                "skew_strength": float(rng.uniform(0.12, 0.35)),
            }
        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def sample_x(self, key: PRNGKeyArray, n: int) -> Array:
        rng = np.random.default_rng(key_to_seed(key))
        if self.feature_mode == "correlated_gaussian":
            mean = np.asarray(self.feature_params["mean"], dtype=np.float64)
            cov = np.asarray(self.feature_params["cov"], dtype=np.float64)
            x = rng.multivariate_normal(mean=mean, cov=cov, size=n)
        elif self.feature_mode == "independent_hetero":
            means = np.asarray(self.feature_params["means"], dtype=np.float64)
            scales = np.asarray(self.feature_params["scales"], dtype=np.float64)
            x = rng.normal(loc=means, scale=scales, size=(n, self.num_features))
            if "near_constant_idx" in self.feature_params:
                idx = np.asarray(self.feature_params["near_constant_idx"], dtype=np.int32)
                x[:, idx] = x[:, idx] * float(self.feature_params["near_constant_scale"])
        else:
            cov = np.asarray(self.feature_params["cov"], dtype=np.float64)
            df = float(self.feature_params["df"])
            x = _sample_multivariate_student_t(n, self.num_features, rng, df, cov)
            skew_idx = np.asarray(self.feature_params["skew_idx"], dtype=np.int32)
            if skew_idx.size > 0:
                x_sub = x[:, skew_idx]
                skew_strength = float(self.feature_params["skew_strength"])
                if self.feature_params["skew_mode"] == "exp":
                    x[:, skew_idx] = x_sub + skew_strength * np.expm1(
                        np.clip(x_sub, -3.0, 3.0)
                    )
                else:
                    x[:, skew_idx] = x_sub + skew_strength * np.log1p(np.exp(x_sub))

        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
        return x.astype(np.float32)

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        x_key, y_key = jax.random.split(key)
        x = np.asarray(self.sample_x(x_key, n), dtype=np.float64)
        rng = np.random.default_rng(key_to_seed(y_key))

        x_info = x[:, self.informative_idx]
        logits = np.zeros((n, self.num_classes), dtype=np.float64)
        for class_idx in range(self.num_classes):
            z = x_info @ self.class_weights[class_idx] + self.class_bias[class_idx]
            for i, j, coef in self.interaction_terms[class_idx]:
                z = z + coef * x_info[:, i] * x_info[:, j]
            logits[:, class_idx] = self.logit_scale * z

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
            "class_counts": {
                int(cls): int((y == cls).sum()) for cls in np.unique(y)
            },
        }
        return {
            "x": x.astype(np.float64),
            "y": y.astype(np.int16),
            "metadata": metadata,
        }
