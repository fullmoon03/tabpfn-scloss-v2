from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from fine_tuning.data import gather_rows, make_eval_indices


def probability_distance(p: Tensor, q: Tensor, distance_name: str = "l1") -> Tensor:
    diff = p - q
    if distance_name == "l1":
        return diff.abs().sum(dim=-1)
    if distance_name == "tv":
        return 0.5 * diff.abs().sum(dim=-1)
    if distance_name == "l2":
        return torch.sqrt((diff * diff).sum(dim=-1).clamp_min(0.0))
    raise ValueError(f"Unsupported probability distance: {distance_name}")


def soft_cross_entropy(target: Tensor, pred: Tensor, eps: float = 1e-12) -> Tensor:
    """Soft-label cross entropy CE(target, pred)."""

    return -(target * torch.log(pred.clamp_min(eps))).sum(dim=-1)


def query_probabilities_from_preprocessed_prefix(
    model: torch.nn.Module,
    *,
    x_context: Tensor,
    y_context: Tensor,
    x_query: Tensor,
    n_classes: int,
    prefix_preprocessor: Any,
) -> Tensor:
    preprocessed_prefix = prefix_preprocessor.fit_prefix(
        x_context.detach().cpu().numpy(),
        y_context.detach().cpu().numpy(),
    )
    if not hasattr(model, "query_probabilities"):
        raise TypeError(
            "preprocessing requires a model with query_probabilities()."
        )
    return model.query_probabilities(
        preprocessed_prefix=preprocessed_prefix,
        x_query=x_query.detach().cpu().numpy(),
        n_classes=n_classes,
    )


def student_rollout_tensors(
    *,
    model: torch.nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    context_indices: Tensor,
    n_classes: int,
    rollout_length: int,
    rollout_seed: int,
    preprocessor_factory: Callable[[], Any],
) -> tuple[Tensor, Tensor]:
    """Generate a hard-label rollout with the current student model.

    The trajectory-generation path is intentionally non-differentiable: labels
    are sampled as discrete integers and the resulting prefixes are treated as
    fixed data for the subsequent differentiable belief/loss computation.
    """

    x_full = gather_rows(x_all, context_indices).detach().clone()
    y_full = gather_rows(y_all, context_indices).detach().clone()
    rng = np.random.default_rng(rollout_seed)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(rollout_length):
                x_idx = int(rng.integers(0, x_full.shape[0]))
                x_new = x_full[x_idx : x_idx + 1]
                probs = query_probabilities_from_preprocessed_prefix(
                    model,
                    x_context=x_full,
                    y_context=y_full,
                    x_query=x_new,
                    n_classes=n_classes,
                    prefix_preprocessor=preprocessor_factory(),
                ).reshape(-1)
                probs_np = probs.detach().cpu().numpy().astype(np.float64)
                probs_np = probs_np / np.clip(probs_np.sum(), 1e-12, None)
                y_new_value = int(rng.choice(np.arange(n_classes), p=probs_np))
                y_new = torch.as_tensor(
                    [y_new_value],
                    dtype=y_full.dtype,
                    device=y_full.device,
                )

                x_full = torch.cat([x_full, x_new.detach()], dim=0)
                y_full = torch.cat([y_full, y_new], dim=0)
    finally:
        model.train(was_training)

    return x_full, y_full


def baseline_query_probabilities_tensor(
    *,
    baseline_pred_rule_factory: Callable[[], Any],
    x_context: Tensor,
    y_context: Tensor,
    x_query: Tensor,
    n_classes: int,
) -> Tensor:
    """Compute frozen baseline p(y | x_query, D_0) as a torch tensor."""

    with torch.no_grad():
        baseline_pred_rule = baseline_pred_rule_factory()
        baseline_pred_rule.fit(
            x_context.detach().cpu().numpy(),
            y_context.detach().cpu().numpy(),
        )
        teacher_np = np.asarray(
            baseline_pred_rule.predict_proba(x_query.detach().cpu().numpy()),
            dtype=np.float64,
        )
    if teacher_np.shape[-1] != n_classes:
        raise ValueError(
            f"Baseline teacher returned {teacher_np.shape[-1]} classes, expected {n_classes}."
        )
    return torch.as_tensor(teacher_np, dtype=x_query.dtype, device=x_query.device)


def validate_batched_indices(idx_context: Tensor, idx_query: Tensor) -> None:
    if idx_query.ndim != 2 or idx_context.ndim != 2:
        raise ValueError("idx_context and idx_query must have shape (batch, n).")
    if idx_query.shape[0] != idx_context.shape[0]:
        raise ValueError("idx_context and idx_query batch dimensions must match.")


def classification_global_emd_loss(
    *,
    model: torch.nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    idx_context: Tensor,
    idx_query: Tensor,
    n_classes: int,
    rollout_length: int,
    rollout_seed: int,
    distance_name: str = "l1",
    preprocessor_factory: Callable[[], Any],
) -> Tensor:
    """Global EMD over queries using student-generated rollout prefixes."""

    validate_batched_indices(idx_context, idx_query)

    batch_losses: list[Tensor] = []
    for batch_idx in range(idx_query.shape[0]):
        prefix_preprocessor = preprocessor_factory()
        context_indices = idx_context[batch_idx]
        x_query = gather_rows(x_all, idx_query[batch_idx])
        x_rollout, y_rollout = student_rollout_tensors(
            model=model,
            x_all=x_all,
            y_all=y_all,
            context_indices=context_indices,
            n_classes=n_classes,
            rollout_length=rollout_length,
            rollout_seed=rollout_seed + batch_idx,
            preprocessor_factory=preprocessor_factory,
        )

        initial_context_size = context_indices.shape[0]
        p0 = query_probabilities_from_preprocessed_prefix(
            model,
            x_context=x_rollout[:initial_context_size],
            y_context=y_rollout[:initial_context_size],
            x_query=x_query,
            n_classes=n_classes,
            prefix_preprocessor=prefix_preprocessor,
        )

        distances: list[Tensor] = []
        for horizon in range(1, rollout_length + 1):
            p_h = query_probabilities_from_preprocessed_prefix(
                model,
                x_context=x_rollout[: initial_context_size + horizon],
                y_context=y_rollout[: initial_context_size + horizon],
                x_query=x_query,
                n_classes=n_classes,
                prefix_preprocessor=prefix_preprocessor,
            )
            distances.append(probability_distance(p_h, p0, distance_name=distance_name))

        emd_per_query = torch.stack(distances, dim=0).mean(dim=0)
        batch_losses.append(emd_per_query.mean())

    return torch.stack(batch_losses).mean()


def classification_self_consistency_loss(
    *,
    model: torch.nn.Module,
    teacher: Tensor,
    x_rollout: Tensor,
    y_rollout: Tensor,
    x_query: Tensor,
    initial_context_size: int,
    n_classes: int,
    eps: float = 1e-12,
    prefix_preprocessor: Any,
) -> Tensor:
    """Differentiable SC loss on an already-fixed rollout trajectory."""

    teacher = teacher.detach()

    prefix_losses: list[Tensor] = []
    for horizon in range(1, x_rollout.shape[0] - initial_context_size + 1):
        student = query_probabilities_from_preprocessed_prefix(
            model,
            x_context=x_rollout[: initial_context_size + horizon],
            y_context=y_rollout[: initial_context_size + horizon],
            x_query=x_query,
            n_classes=n_classes,
            prefix_preprocessor=prefix_preprocessor,
        )
        prefix_losses.append(soft_cross_entropy(teacher, student, eps=eps))

    loss_per_query = torch.stack(prefix_losses, dim=0).mean(dim=0)
    return loss_per_query.mean()


def classification_self_consistency_rollout_objective(
    *,
    model: torch.nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    idx_context: Tensor,
    idx_query: Tensor,
    n_classes: int,
    baseline_pred_rule_factory: Callable[[], Any],
    rollout_length: int,
    rollout_seed: int,
    eps: float = 1e-12,
    preprocessor_factory: Callable[[], Any],
) -> Tensor:
    """SC loss over student-generated rollout prefixes.

    Baseline TabPFN supplies only the stop-gradient D_0 anchor. The current
    student model generates a hard-label rollout without gradient tracking, and
    the differentiable path is the student belief computation on each prefix.
    """

    validate_batched_indices(idx_context, idx_query)

    batch_losses: list[Tensor] = []
    for batch_idx in range(idx_query.shape[0]):
        prefix_preprocessor = preprocessor_factory()
        context_indices = idx_context[batch_idx]
        x_query = gather_rows(x_all, idx_query[batch_idx])
        x_rollout, y_rollout = student_rollout_tensors(
            model=model,
            x_all=x_all,
            y_all=y_all,
            context_indices=context_indices,
            n_classes=n_classes,
            rollout_length=rollout_length,
            rollout_seed=rollout_seed + batch_idx,
            preprocessor_factory=preprocessor_factory,
        )
        initial_context_size = context_indices.shape[0]
        teacher = baseline_query_probabilities_tensor(
            baseline_pred_rule_factory=baseline_pred_rule_factory,
            x_context=x_rollout[:initial_context_size],
            y_context=y_rollout[:initial_context_size],
            x_query=x_query,
            n_classes=n_classes,
        )

        batch_losses.append(
            classification_self_consistency_loss(
                model=model,
                teacher=teacher,
                x_rollout=x_rollout,
                y_rollout=y_rollout,
                x_query=x_query,
                initial_context_size=initial_context_size,
                n_classes=n_classes,
                eps=eps,
                prefix_preprocessor=prefix_preprocessor,
            )
        )

    return torch.stack(batch_losses).mean()


def martingale_loss_fn(
    *,
    loss_name: str = "self_consistency",
    task_type: str,
    model: torch.nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    idx_context: Tensor,
    idx_query: Tensor,
    n_classes: int | None = None,
    baseline_pred_rule_factory: Callable[[], Any] | None = None,
    rollout_length: int = 30,
    rollout_seed: int = 0,
    sc_eps: float = 1e-12,
    preprocessor_factory: Callable[[], Any] | None = None,
) -> Tensor:
    if task_type == "classification":
        if n_classes is None:
            raise ValueError("n_classes is required for classification loss.")
        if baseline_pred_rule_factory is None:
            raise ValueError(
                "baseline_pred_rule_factory is required for classification loss."
            )
        if preprocessor_factory is None:
            raise ValueError(
                "preprocessor_factory is required for classification loss."
            )
        if loss_name == "self_consistency":
            return classification_self_consistency_rollout_objective(
                model=model,
                x_all=x_all,
                y_all=y_all,
                idx_context=idx_context,
                idx_query=idx_query,
                n_classes=n_classes,
                baseline_pred_rule_factory=baseline_pred_rule_factory,
                rollout_length=rollout_length,
                rollout_seed=rollout_seed,
                eps=sc_eps,
                preprocessor_factory=preprocessor_factory,
            )
        raise ValueError(f"Unsupported classification loss_name={loss_name}.")
    if task_type == "regression":
        raise NotImplementedError("Regression martingale loss is not implemented yet.")
    raise ValueError(f"Unsupported task_type={task_type}.")


def martingale_step_fn(
    *,
    model: torch.nn.Module,
    loss_fn: Callable[..., Tensor],
    idx_train: Tensor,
    idx: Tensor,
    state: Any,
    context: dict[str, Any],
) -> Tensor:
    return loss_fn(
        loss_name=context.get("loss_name", "self_consistency"),
        task_type=context.get("task_type", "classification"),
        model=model,
        x_all=context["x_num_train"],
        y_all=context["y_train"],
        idx_context=idx_train,
        idx_query=idx,
        n_classes=context.get("n_classes"),
        baseline_pred_rule_factory=context.get("baseline_pred_rule_factory"),
        rollout_length=context.get("rollout_length", 30),
        rollout_seed=context.get("rollout_seed", 0) + state.step * 1009,
        sc_eps=context.get("sc_eps", 1e-12),
        preprocessor_factory=context.get("preprocessor_factory"),
    )


def evaluate_global_emd(
    *,
    model: torch.nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    query_idx: np.ndarray,
    n_classes: int,
    rollout_length: int,
    rollout_seed: int = 0,
    include_query_in_context: bool = True,
    distance_name: str = "l1",
    preprocessor_factory: Callable[[], Any],
) -> float:
    device = x_all.device
    idx_context, idx_query = make_eval_indices(
        x_all.shape[0],
        query_idx,
        include_query_in_context=include_query_in_context,
        device=device,
    )
    loss = classification_global_emd_loss(
        model=model,
        x_all=x_all,
        y_all=y_all,
        idx_context=idx_context,
        idx_query=idx_query,
        n_classes=n_classes,
        rollout_length=rollout_length,
        rollout_seed=rollout_seed,
        distance_name=distance_name,
        preprocessor_factory=preprocessor_factory,
    )
    return float(loss.detach().cpu())
