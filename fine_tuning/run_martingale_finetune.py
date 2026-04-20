from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import jax
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from dgp import OPENML_CLASSIFICATION, load_dgp
from fine_tuning.data import (
    encode_classification_dataset,
    sample_disjoint_val_test_queries,
)
from fine_tuning.objectives import (
    evaluate_global_emd,
    martingale_loss_fn,
    martingale_step_fn,
)
from fine_tuning.tabpfn_model import TrainableTabPFNClassifier
from fine_tuning.preprocess import make_preprocessor_factory
from fine_tuning.training import (
    EvalResult,
    FullFTConfig,
    train_full_ft,
)
from rollout import TabPFNClassifierPredRule
import utils


def _make_baseline_pred_rule_factory(cfg: DictConfig, categorical_x: list[bool]):
    def factory() -> TabPFNClassifierPredRule:
        return TabPFNClassifierPredRule(
            categorical_x,
            n_estimators=cfg.baseline.n_estimators,
            average_before_softmax=cfg.baseline.average_before_softmax,
            fit_mode="fit_preprocessors",
        )

    return factory


def _make_eval_fn(
    *,
    x_num: torch.Tensor,
    y: torch.Tensor,
    query_idx: np.ndarray,
    n_classes: int,
    baseline_pred_rule_factory,
    rollout_length: int,
    rollout_seed: int,
    include_query_in_context: bool,
    distance_name: str,
    preprocessor_factory,
):
    def eval_fn(*, model: torch.nn.Module, state, context) -> EvalResult:
        val_emd = evaluate_global_emd(
            model=model,
            x_all=x_num,
            y_all=y,
            query_idx=query_idx,
            n_classes=n_classes,
            baseline_pred_rule_factory=baseline_pred_rule_factory,
            rollout_length=rollout_length,
            rollout_seed=rollout_seed,
            include_query_in_context=include_query_in_context,
            distance_name=distance_name,
            preprocessor_factory=preprocessor_factory,
        )
        return EvalResult(score=-val_emd, metrics={"val_global_emd": val_emd})

    return eval_fn


def run(cfg: DictConfig) -> None:
    if cfg.task_type != "classification":
        raise NotImplementedError("Only classification fine-tuning is implemented.")
    if cfg.dgp.name not in OPENML_CLASSIFICATION:
        raise NotImplementedError(
            "Martingale fine-tuning split is currently implemented only for "
            f"OPENML_CLASSIFICATION datasets, got {cfg.dgp.name}."
        )

    outdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(outdir, exist_ok=True)

    data_key = jax.random.key(cfg.seed * 37)
    dgp = load_dgp(cfg, data_key)
    train_data = dgp.train_data
    val_test_data = dgp.test_data

    if train_data["x"].shape[0] != cfg.data_size:
        raise ValueError(
            f"Expected train size {cfg.data_size}, got {train_data['x'].shape[0]}."
        )
    if val_test_data["x"].shape[0] != cfg.val_test_size:
        raise ValueError(
            f"Expected val/test pool size {cfg.val_test_size}, got {val_test_data['x'].shape[0]}."
        )
    if any(getattr(dgp, "categorical_x", [])):
        raise NotImplementedError(
            "Categorical OpenML features are not wired into this fine-tuning runner yet."
        )
    baseline_pred_rule_factory = _make_baseline_pred_rule_factory(cfg, dgp.categorical_x)
    categorical_features_indices = [
        idx for idx, is_categorical in enumerate(dgp.categorical_x) if is_categorical
    ]

    device = torch.device(
        cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    preprocessor_factory = make_preprocessor_factory(
        categorical_features_indices=categorical_features_indices,
        n_estimators=cfg.baseline.n_estimators,
        average_before_softmax=cfg.baseline.average_before_softmax,
        model_path=cfg.checkpoint_path,
        device="cpu",
    )
    class_labels = np.unique(np.concatenate([train_data["y"], val_test_data["y"]]))
    train_encoded = encode_classification_dataset(
        train_data["x"],
        train_data["y"],
        class_labels=class_labels,
        device=device,
    )
    val_test_encoded = encode_classification_dataset(
        val_test_data["x"],
        val_test_data["y"],
        class_labels=class_labels,
        device=device,
    )

    query_split = sample_disjoint_val_test_queries(
        pool_size=cfg.val_test_size,
        query_size=cfg.query_size,
        seed=cfg.query_seed,
    )
    utils.write_to(
        f"{outdir}/split.pickle",
        {
            "train_size": cfg.data_size,
            "train_step_context_size": cfg.context_size,
            "val_test_size": cfg.val_test_size,
            "class_labels": class_labels,
            "val_query_idx": query_split.val_query_idx,
            "test_query_idx": query_split.test_query_idx,
            "val_query_y": val_test_data["y"][query_split.val_query_idx],
            "test_query_y": val_test_data["y"][query_split.test_query_idx],
            "query_source": cfg.query_source,
            "include_query_in_context": cfg.include_query_in_context,
            "training_loss": cfg.loss.name,
        },
        verbose=True,
    )

    model = TrainableTabPFNClassifier(
        checkpoint_path=Path(cfg.checkpoint_path),
        n_estimators=cfg.baseline.n_estimators,
        average_before_softmax=cfg.baseline.average_before_softmax,
        categorical_features_indices=categorical_features_indices,
        device=device,
    )

    ft_config = FullFTConfig(
        train_size=cfg.data_size,
        n_steps=cfg.n_epochs * cfg.epoch_size,
        epoch_size=cfg.epoch_size,
        batch_size=cfg.batch_size,
        context_size=cfg.context_size,
        query_size=cfg.query_size,
        patience=cfg.patience,
        optimizer=dict(cfg.optimizer),
        gradient_clipping_norm=cfg.gradient_clipping_norm,
        n_lr_warmup_epochs=cfg.n_lr_warmup_epochs,
        amp=cfg.amp,
        randperm=cfg.randperm,
        seed=cfg.seed,
        output_dir=outdir,
        save_best=cfg.save_best,
        show_progress=cfg.show_progress,
        progress_log_every_epochs=cfg.progress_log_every_epochs,
    )

    context = {
        "task_type": cfg.task_type,
        "loss_name": cfg.loss.name,
        "x_num_train": train_encoded.x_num,
        "y_train": train_encoded.y,
        "n_classes": len(class_labels),
        "baseline_pred_rule_factory": baseline_pred_rule_factory,
        "preprocessor_factory": preprocessor_factory,
        "rollout_length": cfg.rollout_length,
        "rollout_seed": cfg.rollout_seed,
        "include_query_in_context": cfg.include_query_in_context,
        "sc_eps": cfg.loss.sc_eps,
    }
    eval_fn = _make_eval_fn(
        x_num=val_test_encoded.x_num,
        y=val_test_encoded.y,
        query_idx=query_split.val_query_idx,
        n_classes=len(class_labels),
        baseline_pred_rule_factory=baseline_pred_rule_factory,
        rollout_length=cfg.rollout_length,
        rollout_seed=cfg.rollout_seed + 50_000,
        include_query_in_context=cfg.include_query_in_context,
        distance_name=cfg.metrics.emd.distance,
        preprocessor_factory=preprocessor_factory,
    )

    state = train_full_ft(
        model=model,
        config=ft_config,
        step_fn=martingale_step_fn,
        loss_fn=martingale_loss_fn,
        eval_fn=eval_fn,
        context=context,
        device=device,
    )

    model.eval()
    with torch.inference_mode():
        val_emd = evaluate_global_emd(
            model=model,
            x_all=val_test_encoded.x_num,
            y_all=val_test_encoded.y,
            query_idx=query_split.val_query_idx,
            n_classes=len(class_labels),
            baseline_pred_rule_factory=baseline_pred_rule_factory,
            rollout_length=cfg.rollout_length,
            rollout_seed=cfg.rollout_seed + 50_000,
            include_query_in_context=cfg.include_query_in_context,
            distance_name=cfg.metrics.emd.distance,
            preprocessor_factory=preprocessor_factory,
        )
        test_emd = evaluate_global_emd(
            model=model,
            x_all=val_test_encoded.x_num,
            y_all=val_test_encoded.y,
            query_idx=query_split.test_query_idx,
            n_classes=len(class_labels),
            baseline_pred_rule_factory=baseline_pred_rule_factory,
            rollout_length=cfg.rollout_length,
            rollout_seed=cfg.rollout_seed + 100_000,
            include_query_in_context=cfg.include_query_in_context,
            distance_name=cfg.metrics.emd.distance,
            preprocessor_factory=preprocessor_factory,
        )

    utils.write_to(
        f"{outdir}/training-state.pickle",
        {
            "state": state,
            "history": state.history,
            "best_score": state.best_score,
            "best_step": state.best_step,
            "final_val_global_emd": val_emd,
            "final_test_global_emd": test_emd,
        },
        verbose=True,
    )
    logging.info("Final val Global EMD: %.6f", val_emd)
    logging.info("Final test Global EMD: %.6f", test_emd)


@hydra.main(version_base=None, config_path="../conf", config_name="martingale-finetune")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    utils.suppress_noisy_third_party_logs()
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    utils.suppress_noisy_third_party_logs()
    OmegaConf.register_new_resolver("githash", utils.githash)
    OmegaConf.register_new_resolver("kst_hhmm", utils.kst_hhmm)
    main()
