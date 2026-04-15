import logging
import os
import warnings

import hydra
import jax
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import utils
from dgp import OPENML_BINARY_CLASSIFICATION, OPENML_CLASSIFICATION, load_dgp
from rollout import TabPFNClassifierPredRule, get_x_new

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def make_classifier_pred_rule(cfg: DictConfig, dgp: object) -> TabPFNClassifierPredRule:
    dim_x = dgp.train_data["x"].shape[-1]
    if cfg.dgp.name.startswith("classification-fixed") or cfg.dgp.name == "classification-scm":
        categorical_x = [False] * dim_x
    elif cfg.dgp.name in OPENML_CLASSIFICATION + OPENML_BINARY_CLASSIFICATION:
        categorical_x = dgp.categorical_x
    else:
        raise ValueError(
            f"Fixed-query belief trajectory currently supports classification datasets only, got {cfg.dgp.name}."
        )
    return TabPFNClassifierPredRule(
        categorical_x, cfg.n_estimators, cfg.average_before_softmax
    )


def sample_test_queries(
    test_data: dict[str, np.ndarray], num_queries: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_test = test_data["x"].shape[0]
    if n_test < num_queries:
        raise ValueError(f"Need at least {num_queries} test queries, found {n_test}.")

    rng = np.random.default_rng(seed)
    query_idx = np.sort(rng.choice(n_test, size=num_queries, replace=False))
    return query_idx, test_data["x"][query_idx], test_data["y"][query_idx]


def query_belief(
    pred_rule: TabPFNClassifierPredRule,
    x_query: np.ndarray,
) -> np.ndarray:
    return pred_rule.predict_proba(x_query)


def single_rollout_belief_trajectory(
    key: jax.Array,
    pred_rule: TabPFNClassifierPredRule,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    rollout_length: int,
) -> np.ndarray:
    x_context = np.array(x_train, copy=True)
    y_context = np.array(y_train, copy=True)

    pred_rule.fit(x_context, y_context)
    init_belief = query_belief(pred_rule, x_query)
    beliefs = np.empty(
        (rollout_length + 1, x_query.shape[0], init_belief.shape[1]), dtype=np.float64
    )
    beliefs[0] = init_belief

    for depth in range(1, rollout_length + 1):
        rkey = jax.random.fold_in(key, depth)
        rkey, subkey = jax.random.split(rkey)
        x_new = np.asarray(get_x_new(subkey, x_context))

        rkey, subkey = jax.random.split(rkey)
        y_new = pred_rule.sample(subkey, x_new, x_context, y_context)

        x_context = np.concatenate([x_context, x_new], axis=0)
        y_context = np.concatenate([y_context, np.atleast_1d(y_new)], axis=0)

        pred_rule.fit(x_context, y_context)
        beliefs[depth] = query_belief(pred_rule, x_query)

    return beliefs


def plot_mean_belief(
    mean_belief: np.ndarray,
    query_indices: np.ndarray,
    class_labels: np.ndarray,
    queries_per_figure: int,
    plot_rows: int,
    plot_cols: int,
    outdir: str,
    dataset_name: str,
) -> None:
    depth = np.arange(mean_belief.shape[0])
    num_queries = mean_belief.shape[1]
    num_classes = mean_belief.shape[2]
    class_handles = None

    for fig_idx, start in enumerate(range(0, num_queries, queries_per_figure), start=1):
        end = min(start + queries_per_figure, num_queries)
        fig, axes = plt.subplots(
            plot_rows, plot_cols, figsize=(plot_cols * 4.0, plot_rows * 3.0), sharex=True, sharey=True
        )
        axes = np.asarray(axes).reshape(-1)

        for ax_idx, query_pos in enumerate(range(start, end)):
            ax = axes[ax_idx]
            line_handles = []
            for class_idx in range(num_classes):
                (line,) = ax.plot(
                    depth,
                    mean_belief[:, query_pos, class_idx],
                    linewidth=1.5,
                    marker="o",
                    markersize=2.2,
                    label=f"class {class_idx} ({class_labels[class_idx]})",
                )
                line_handles.append(line)
            if class_handles is None:
                class_handles = line_handles

            ax.set_title(f"q_idx={query_indices[query_pos]}")
            ax.set_xlim(depth[0], depth[-1])
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)

        for ax in axes[end - start :]:
            ax.axis("off")

        fig.suptitle(
            f"{dataset_name.capitalize()} fixed-query mean belief across paths "
            f"({start + 1}-{end}/{num_queries})",
            y=0.995,
        )
        fig.supxlabel("Depth n")
        fig.supylabel("Mean Belief")
        if class_handles is not None:
            fig.legend(
                class_handles,
                [handle.get_label() for handle in class_handles],
                loc="upper right",
                frameon=True,
            )
        fig.tight_layout(rect=(0, 0, 0.97, 0.97))
        fig.savefig(f"{outdir}/belief-trajectory-{fig_idx}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


@hydra.main(version_base=None, config_path="conf", config_name="fixed-query-belief")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    outdir = os.path.relpath(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.makedirs(outdir, exist_ok=True)

    torch.manual_seed(cfg.seed * 71)
    base_key = jax.random.key(cfg.seed * 37)
    base_key, data_key = jax.random.split(base_key)

    dgp = load_dgp(cfg, data_key)
    if getattr(dgp, "test_data", None) is None:
        raise ValueError(f"{cfg.dgp.name} does not provide a held-out test split for query sampling.")

    query_idx, x_query, y_query = sample_test_queries(dgp.test_data, cfg.num_queries, cfg.seed * 101)
    utils.write_to(
        f"{outdir}/queries.pickle",
        {"idx": query_idx, "x": x_query, "y": y_query},
        verbose=True,
    )

    pred_rule = make_classifier_pred_rule(cfg, dgp)
    pred_rule.fit(dgp.train_data["x"], dgp.train_data["y"])
    class_labels = np.asarray(pred_rule.classes_)

    beliefs = np.empty(
        (
            cfg.rollout_times,
            cfg.rollout_length + 1,
            cfg.num_queries,
            class_labels.shape[0],
        ),
        dtype=np.float64,
    )

    for rollout_idx in tqdm(range(cfg.rollout_times), desc="Rollouts"):
        pred_rule = make_classifier_pred_rule(cfg, dgp)
        rollout_key = jax.random.fold_in(base_key, rollout_idx)
        beliefs[rollout_idx] = single_rollout_belief_trajectory(
            rollout_key,
            pred_rule,
            dgp.train_data["x"],
            dgp.train_data["y"],
            x_query,
            cfg.rollout_length,
        )

    mean_belief = beliefs.mean(axis=0)
    utils.write_to(
        f"{outdir}/belief-trajectory.pickle",
        {
            "beliefs": beliefs,
            "mean_belief": mean_belief,
            "query_idx": query_idx,
            "x_query": x_query,
            "y_query": y_query,
            "class_labels": class_labels,
        },
        verbose=True,
    )
    plot_mean_belief(
        mean_belief,
        query_idx,
        class_labels,
        cfg.queries_per_figure,
        cfg.plot_rows,
        cfg.plot_cols,
        outdir,
        cfg.dgp.name,
    )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
