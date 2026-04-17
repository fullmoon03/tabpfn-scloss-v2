from __future__ import annotations

import jax
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .base import SyntheticGenerator


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
