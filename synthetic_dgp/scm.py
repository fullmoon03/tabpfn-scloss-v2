from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .base import SyntheticGenerator, key_to_seed


ActivationName = str


@dataclass
class SCMHyperparameters:
    num_features: int
    num_classes: int
    num_causes: int
    num_layers: int
    hidden_dim: int
    noise_std: float
    init_std: float
    is_causal: bool
    y_is_effect: bool
    block_wise_dropout: bool
    sort_features: bool
    in_clique: bool
    random_feature_rotation: bool
    activation: ActivationName
    sampling: str


@dataclass
class SCMParameters:
    hyperparameters: SCMHyperparameters
    cause_specs: list[dict[str, Any]]
    layer_weights: list[np.ndarray]
    class_weights: np.ndarray
    class_bias: np.ndarray
    feature_idx: np.ndarray
    rotation_shift: int


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _get_activation(name: ActivationName):
    if name == "tanh":
        return _tanh
    if name == "relu":
        return _relu
    if name == "identity":
        return _identity
    raise ValueError(f"Unsupported activation: {name}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _sample_positive_lognormal(
    rng: np.random.Generator,
    min_mean: float,
    max_mean: float,
    min_std: float = 0.01,
    max_std: float = 1.0,
) -> float:
    log_mean = rng.uniform(np.log(min_mean), np.log(max_mean))
    log_std = rng.uniform(np.log(min_std), np.log(max_std))
    mean = np.exp(log_mean)
    std = mean * np.exp(log_std)
    value = rng.normal(mean, std)
    return float(max(0.0, value))


def _sample_hyperparameters(
    rng: np.random.Generator,
    num_features: int,
    num_classes: int,
) -> SCMHyperparameters:
    num_causes = int(
        max(
            num_features + 1,
            2
            + round(
                rng.gamma(
                    shape=rng.uniform(0.5, 3.0),
                    scale=rng.uniform(0.5, 2.5),
                )
            ),
        )
    )
    num_layers = int(
        max(
            2,
            2
            + round(
                rng.gamma(
                    shape=rng.uniform(0.5, 2.0),
                    scale=rng.uniform(0.5, 1.5),
                )
            ),
        )
    )
    hidden_dim = int(
        max(
            num_features + num_classes,
            4
            + round(
                rng.gamma(
                    shape=rng.uniform(0.5, 3.0),
                    scale=rng.uniform(2.0, 12.0),
                )
            ),
        )
    )

    return SCMHyperparameters(
        num_features=num_features,
        num_classes=num_classes,
        num_causes=num_causes,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        noise_std=_sample_positive_lognormal(rng, min_mean=1e-4, max_mean=0.3),
        init_std=max(
            1e-4,
            _sample_positive_lognormal(rng, min_mean=0.01, max_mean=3.0),
        ),
        is_causal=bool(rng.integers(0, 2)),
        y_is_effect=bool(rng.integers(0, 2)),
        block_wise_dropout=bool(rng.integers(0, 2)),
        sort_features=bool(rng.integers(0, 2)),
        in_clique=bool(rng.integers(0, 2)),
        random_feature_rotation=bool(rng.integers(0, 2)),
        activation=rng.choice(["tanh", "relu", "identity"]).item(),
        sampling=rng.choice(["normal", "mixed"], p=[0.7, 0.3]).item(),
    )


def _sample_cause_specs(
    rng: np.random.Generator, num_causes: int, sampling: str
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for _ in range(num_causes):
        if sampling == "normal":
            kind = "normal"
        elif sampling == "mixed":
            kind = rng.choice(
                ["normal", "categorical", "zipf"], p=[0.45, 0.35, 0.20]
            ).item()
        else:
            raise ValueError(f"Unsupported sampling mode: {sampling}")

        if kind == "normal":
            mean = float(rng.normal(0.0, 1.0))
            std = float(max(np.abs(rng.normal(0.0, 1.0)), 0.05))
            specs.append({"kind": kind, "mean": mean, "std": std})
        elif kind == "categorical":
            n_categories = int(rng.integers(2, 11))
            probs = rng.random(n_categories)
            probs = probs / probs.sum()
            specs.append({"kind": kind, "probs": probs.astype(np.float32)})
        else:
            exponent = float(rng.uniform(2.0, 4.0))
            specs.append({"kind": kind, "a": exponent})
    return specs


def _sample_causes(
    rng: np.random.Generator,
    num_samples: int,
    cause_specs: list[dict[str, Any]],
) -> np.ndarray:
    cols = []
    for spec in cause_specs:
        if spec["kind"] == "normal":
            col = rng.normal(spec["mean"], spec["std"], size=num_samples)
        elif spec["kind"] == "categorical":
            probs = np.asarray(spec["probs"], dtype=np.float64)
            col = rng.choice(np.arange(probs.size), size=num_samples, p=probs)
            col = col.astype(np.float32)
            col = (col - col.mean()) / max(col.std(), 1e-6)
        elif spec["kind"] == "zipf":
            col = np.minimum(rng.zipf(a=spec["a"], size=num_samples), 10).astype(
                np.float32
            )
            col = col - col.mean()
        else:
            raise ValueError(f"Unsupported cause spec: {spec['kind']}")
        cols.append(np.asarray(col, dtype=np.float32))
    return np.stack(cols, axis=1)


def _make_weight(
    rng: np.random.Generator,
    in_dim: int,
    out_dim: int,
    init_std: float,
    block_wise_dropout: bool,
) -> np.ndarray:
    if block_wise_dropout and min(in_dim, out_dim) >= 2:
        weight = np.zeros((in_dim, out_dim), dtype=np.float32)
        n_blocks = int(
            rng.integers(
                1, max(2, int(np.ceil(np.sqrt(min(in_dim, out_dim)))) + 1)
            )
        )
        row_edges = np.linspace(0, in_dim, n_blocks + 1, dtype=int)
        col_edges = np.linspace(0, out_dim, n_blocks + 1, dtype=int)
        for block in range(n_blocks):
            r0, r1 = row_edges[block], row_edges[block + 1]
            c0, c1 = col_edges[block], col_edges[block + 1]
            if r1 > r0 and c1 > c0:
                weight[r0:r1, c0:c1] = rng.normal(
                    0.0, init_std, size=(r1 - r0, c1 - c0)
                )
        if np.any(weight):
            return weight.astype(np.float32)

    dropout_prob = float(min(rng.beta(2.0, 5.0) * 0.6, 0.99))
    weight = rng.normal(0.0, init_std, size=(in_dim, out_dim)).astype(np.float32)
    mask = rng.binomial(1, 1.0 - dropout_prob, size=(in_dim, out_dim)).astype(
        np.float32
    )
    return (weight * mask).astype(np.float32)


def _forward_hidden(
    rng: np.random.Generator,
    causes: np.ndarray,
    params: SCMParameters,
) -> tuple[list[np.ndarray], np.ndarray]:
    hp = params.hyperparameters
    activation = _get_activation(hp.activation)

    current = causes
    outputs = []
    for layer_idx, weight in enumerate(params.layer_weights):
        current = current @ weight
        if layer_idx > 0 and hp.noise_std > 0.0:
            current = current + rng.normal(
                0.0, hp.noise_std, size=current.shape
            ).astype(np.float32)
        outputs.append(current.astype(np.float32))
        if layer_idx < len(params.layer_weights) - 1:
            current = activation(current)

    hidden = outputs[-1]
    if hp.is_causal:
        node_matrix = np.concatenate(outputs, axis=1)
    else:
        node_matrix = hidden
    return outputs, node_matrix


def _rotate_features(
    x: np.ndarray, random_feature_rotation: bool, rotation_shift: int
) -> np.ndarray:
    if not random_feature_rotation or x.shape[1] <= 1:
        return x
    return np.roll(x, shift=rotation_shift, axis=1)


def _normalize_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def sample_scm_parameters(
    num_features: int,
    num_classes: int,
    seed: int | None = None,
) -> SCMParameters:
    if num_features < 1:
        raise ValueError("num_features must be at least 1")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    rng = np.random.default_rng(seed)
    hp = _sample_hyperparameters(rng, num_features=num_features, num_classes=num_classes)
    cause_specs = _sample_cause_specs(rng, hp.num_causes, hp.sampling)

    layer_weights = [
        _make_weight(
            rng,
            in_dim=hp.num_causes if layer_idx == 0 else hp.hidden_dim,
            out_dim=hp.hidden_dim,
            init_std=hp.init_std,
            block_wise_dropout=hp.block_wise_dropout,
        )
        for layer_idx in range(hp.num_layers)
    ]

    total_nodes = hp.hidden_dim * hp.num_layers if hp.is_causal else hp.hidden_dim
    if hp.in_clique:
        start_max = max(1, total_nodes - hp.num_features + 1)
        start = int(rng.integers(0, start_max))
        feature_idx = np.arange(start, start + hp.num_features)
    else:
        feature_idx = rng.choice(total_nodes, size=hp.num_features, replace=False)

    if hp.sort_features:
        feature_idx = np.sort(feature_idx)

    logits_input_dim = total_nodes if hp.y_is_effect else hp.num_causes
    class_weights = rng.normal(
        0.0, hp.init_std, size=(logits_input_dim, hp.num_classes)
    ).astype(np.float32)
    class_bias = rng.normal(0.0, hp.init_std, size=(hp.num_classes,)).astype(np.float32)

    return SCMParameters(
        hyperparameters=hp,
        cause_specs=cause_specs,
        layer_weights=layer_weights,
        class_weights=class_weights,
        class_bias=class_bias,
        feature_idx=np.asarray(feature_idx, dtype=np.int32),
        rotation_shift=int(rng.integers(0, hp.num_features)),
    )


def generate_scm_classification_dataset(
    num_samples: int,
    parameters: SCMParameters,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    rng = np.random.default_rng(seed)
    hp = parameters.hyperparameters

    causes = _sample_causes(rng, num_samples=num_samples, cause_specs=parameters.cause_specs)
    _, node_matrix = _forward_hidden(rng, causes, parameters)

    x = node_matrix[:, parameters.feature_idx].astype(np.float32)
    x = _rotate_features(x, hp.random_feature_rotation, parameters.rotation_shift)
    x = _normalize_features(x)

    logits_input = node_matrix if hp.y_is_effect else causes
    logits = logits_input @ parameters.class_weights + parameters.class_bias
    if hp.noise_std > 0.0:
        logits = logits + rng.normal(0.0, hp.noise_std, size=logits.shape).astype(
            np.float32
        )
    probs = _softmax(logits.astype(np.float64)).astype(np.float32)
    y = np.array(
        [rng.choice(hp.num_classes, p=prob) for prob in probs], dtype=np.int64
    )

    metadata = {
        "hyperparameters": asdict(hp),
        "num_samples": int(num_samples),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "class_counts": {int(cls): int((y == cls).sum()) for cls in np.unique(y)},
        "feature_idx": parameters.feature_idx.tolist(),
    }
    return x.astype(np.float32), y, metadata


class SCMClassificationGenerator(SyntheticGenerator):
    num_classes: int
    parameters: SCMParameters

    def __init__(self, num_features: int, num_classes: int, seed: int | None = None):
        super().__init__(num_features)
        self.num_classes = num_classes
        self.parameters = sample_scm_parameters(
            num_features=num_features,
            num_classes=num_classes,
            seed=seed,
        )
        self.metadata = {"hyperparameters": self.parameters.hyperparameters}

    def sample_x(self, key: PRNGKeyArray, n: int) -> Array:
        return self.sample(key, n)["x"]

    def sample(self, key: PRNGKeyArray, n: int) -> dict[str, object]:
        seed = key_to_seed(key)
        x, y, metadata = generate_scm_classification_dataset(
            num_samples=n,
            parameters=self.parameters,
            seed=seed,
        )
        return {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.int16),
            "metadata": metadata,
        }


def save_dataset(
    output_path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X=x, y=y, metadata=json.dumps(metadata))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an SCM synthetic classification dataset."
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional .npz output path."
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    params = sample_scm_parameters(
        num_features=args.num_features,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    x, y, metadata = generate_scm_classification_dataset(
        num_samples=args.num_samples,
        parameters=params,
        seed=args.seed + 1,
    )

    if args.output is not None:
        save_dataset(args.output, x, y, metadata)
        print(f"saved={args.output}")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
