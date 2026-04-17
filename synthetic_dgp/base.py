from __future__ import annotations

from abc import ABC, abstractmethod

import jax
from jaxtyping import Array, PRNGKeyArray


def key_to_seed(key: PRNGKeyArray) -> int:
    return int(
        jax.random.randint(key, shape=(), minval=0, maxval=2_147_483_647).item()
    )


class SyntheticGenerator(ABC):
    num_features: int
    target_name: str
    feature_name: list[str]
    categorical_x: list[bool]
    metadata: dict[str, object]

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.target_name = "y"
        self.feature_name = [f"x{i + 1}" for i in range(num_features)]
        self.categorical_x = [False] * num_features
        self.metadata = {}

    @abstractmethod
    def sample_x(self, key: PRNGKeyArray, n: int) -> Array:
        pass

    @abstractmethod
    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        pass
