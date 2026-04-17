from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .base import SyntheticGenerator


class _WuMartinGenerator(SyntheticGenerator):
    beta0: np.ndarray

    def __init__(self):
        beta0 = np.asarray([1.0, 1.0, 2.0, -1.0], dtype=np.float64)
        super().__init__(beta0.shape[0])
        self.beta0 = beta0

    def sample_x(self, key: PRNGKeyArray, n: int) -> Array:
        rho = 0.2
        key, x_key = jax.random.split(key)
        indices = jnp.arange(self.num_features)
        idx_diff = jnp.abs(indices[:, None] - indices[None, :])
        corr_matrix = rho**idx_diff
        return jax.random.multivariate_normal(
            x_key, mean=jnp.zeros(self.num_features), cov=corr_matrix, shape=(n,)
        )


class LinearRegressionWMGenerator(_WuMartinGenerator):
    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        y = x @ self.beta0 + jax.random.normal(y_key, shape=(n,))
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
        }


class DependentErrorWMGenerator(_WuMartinGenerator):
    s_small: float
    s_mod: float

    def __init__(self, s_small: float, s_mod: float):
        super().__init__()
        self.s_small = s_small
        self.s_mod = s_mod

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        x_05 = jnp.quantile(x[:, 0], 0.05, axis=0)
        x_95 = jnp.quantile(x[:, 0], 0.95, axis=0)
        std = jnp.where(
            x[:, 0] < x_05,
            self.s_small,
            jnp.where(x[:, 0] < x_95, self.s_mod, 1),
        )
        mean = x @ self.beta0
        y = mean + std * jax.random.normal(y_key, shape=(n,))
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
        }


class NonNormalErrorWMGenerator(_WuMartinGenerator):
    df: int

    def __init__(self, df: int):
        super().__init__()
        self.df = df

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        mean = x @ self.beta0
        y = mean + jax.random.t(y_key, df=self.df, shape=(n,))
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
        }
