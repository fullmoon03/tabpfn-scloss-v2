from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .base import SyntheticGenerator


class _LinearRegressionGenerator(SyntheticGenerator):
    beta0: np.ndarray

    def __init__(self, dim_x: int, beta0: np.ndarray):
        super().__init__(dim_x)
        self.beta0 = np.asarray(beta0, dtype=np.float64)

    def sample_x(self, key: PRNGKeyArray, n: int) -> Array:
        key, subkey = jax.random.split(key)
        return jax.random.uniform(subkey, shape=(n, self.num_features), minval=-1, maxval=1)


class RegressionFixedGenerator(_LinearRegressionGenerator):
    noise_std: float

    def __init__(self, dim_x: int, noise_std: float):
        fixed_key = jax.random.key(1058)
        beta0 = jax.random.uniform(fixed_key, shape=(dim_x,), minval=-2, maxval=3)
        super().__init__(dim_x, np.asarray(beta0, dtype=np.float64))
        self.noise_std = noise_std

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        y = x @ self.beta0 + jax.random.normal(y_key, shape=(n,)) * self.noise_std
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
        }


class RegressionFixedDependentErrorGenerator(_LinearRegressionGenerator):
    s_small: float
    s_mod: float

    def __init__(self, dim_x: int, s_small: float, s_mod: float):
        fixed_key = jax.random.key(1058)
        beta0 = jax.random.uniform(fixed_key, shape=(dim_x,), minval=-2, maxval=3)
        super().__init__(dim_x, np.asarray(beta0, dtype=np.float64))
        self.s_small = s_small
        self.s_mod = s_mod

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        x_lower = jnp.quantile(x[:, 0], 0.25, axis=0)
        x_upper = jnp.quantile(x[:, 0], 0.75, axis=0)
        std = jnp.where(
            x[:, 0] < x_lower,
            self.s_small,
            jnp.where(x[:, 0] < x_upper, self.s_mod, 1),
        )
        mean = x @ self.beta0
        y = mean + std * jax.random.normal(y_key, shape=(n,))
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
        }


class RegressionFixedNonNormalErrorGenerator(_LinearRegressionGenerator):
    df: int

    def __init__(self, dim_x: int, df: int):
        fixed_key = jax.random.key(1058)
        beta0 = jax.random.uniform(fixed_key, shape=(dim_x,), minval=-2, maxval=3)
        super().__init__(dim_x, np.asarray(beta0, dtype=np.float64))
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


class RegressionPriorGenerator(_LinearRegressionGenerator):
    noise_std0: float

    def __init__(self, key: PRNGKeyArray, dim_x: int):
        beta0 = jax.random.normal(key, shape=(dim_x,))
        super().__init__(dim_x, np.asarray(beta0, dtype=np.float64))
        self.noise_std0 = math.sqrt(0.1)

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        key, x_key, y_key = jax.random.split(key, 3)
        x = self.sample_x(x_key, n)
        error = jax.random.normal(y_key, shape=(n,)) * self.noise_std0
        y = x @ self.beta0 + error
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
        }
