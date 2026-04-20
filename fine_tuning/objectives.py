from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
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


def baseline_rollout_numpy(
    *,
    baseline_pred_rule_factory: Callable[[], Any],
    x_context: np.ndarray,
    y_context: np.ndarray,
    rollout_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a fixed discrete rollout using a frozen baseline TabPFN model."""

    x_full = np.asarray(x_context, dtype=np.float64).copy()
    y_full = np.asarray(y_context).copy()
    rng = np.random.default_rng(seed)
    baseline_pred_rule = baseline_pred_rule_factory()

    for _ in range(rollout_length):
        x_idx = rng.integers(0, x_full.shape[0])
        x_new = x_full[x_idx : x_idx + 1]

        # The baseline model and categorical sampling are intentionally outside
        # the training graph. The returned trajectory is treated as fixed data.
        baseline_pred_rule.fit(x_full, y_full)
        probs = np.asarray(baseline_pred_rule.predict_proba(x_new)).reshape(-1)
        probs = probs / probs.sum()
        y_new = rng.choice(np.asarray(baseline_pred_rule.classes_), p=probs)

        x_full = np.concatenate([x_full, x_new], axis=0)
        y_full = np.concatenate([y_full, np.asarray([y_new], dtype=y_full.dtype)])

    return x_full, y_full


def query_probabilities_from_fixed_context(
    model: nn.Module,
    *,
    x_context: Tensor,
    y_context: Tensor,
    x_query: Tensor,
    n_classes: int,
) -> Tensor:
    x_num = torch.cat([x_context[None, :, :], x_query[None, :, :]], dim=1)
    logits = model(x_num=x_num, y_train=y_context[None, :])[..., :n_classes]
    return logits.softmax(dim=-1).squeeze(0)


def fixed_baseline_rollout_tensors(
    *,
    baseline_pred_rule_factory: Callable[[], Any],
    x_all: Tensor,
    y_all: Tensor,
    context_indices: Tensor,
    rollout_length: int,
    rollout_seed: int,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        x_context_np = gather_rows(x_all, context_indices).detach().cpu().numpy()
        y_context_np = gather_rows(y_all, context_indices).detach().cpu().numpy()
        x_rollout_np, y_rollout_np = baseline_rollout_numpy(
            baseline_pred_rule_factory=baseline_pred_rule_factory,
            x_context=x_context_np,
            y_context=y_context_np,
            rollout_length=rollout_length,
            seed=rollout_seed,
        )

    return (
        torch.as_tensor(x_rollout_np, dtype=x_all.dtype, device=x_all.device),
        torch.as_tensor(y_rollout_np, dtype=y_all.dtype, device=y_all.device),
    )


def validate_batched_indices(idx_context: Tensor, idx_query: Tensor) -> None:
    if idx_query.ndim != 2 or idx_context.ndim != 2:
        raise ValueError("idx_context and idx_query must have shape (batch, n).")
    if idx_query.shape[0] != idx_context.shape[0]:
        raise ValueError("idx_context and idx_query batch dimensions must match.")


def classification_global_emd_loss(
    *,
    model: nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    idx_context: Tensor,
    idx_query: Tensor,
    n_classes: int,
    baseline_pred_rule_factory: Callable[[], Any],
    rollout_length: int,
    rollout_seed: int,
    distance_name: str = "l1",
) -> Tensor:
    """Global EMD over queries using frozen baseline rollout prefixes."""

    validate_batched_indices(idx_context, idx_query)

    batch_losses: list[Tensor] = []
    for batch_idx in range(idx_query.shape[0]):
        context_indices = idx_context[batch_idx]
        x_query = gather_rows(x_all, idx_query[batch_idx])
        x_rollout, y_rollout = fixed_baseline_rollout_tensors(
            baseline_pred_rule_factory=baseline_pred_rule_factory,
            x_all=x_all,
            y_all=y_all,
            context_indices=context_indices,
            rollout_length=rollout_length,
            rollout_seed=rollout_seed + batch_idx,
        )

        initial_context_size = context_indices.shape[0]
        p0 = query_probabilities_from_fixed_context(
            model,
            x_context=x_rollout[:initial_context_size],
            y_context=y_rollout[:initial_context_size],
            x_query=x_query,
            n_classes=n_classes,
        )

        distances: list[Tensor] = []
        for horizon in range(1, rollout_length + 1):
            p_h = query_probabilities_from_fixed_context(
                model,
                x_context=x_rollout[: initial_context_size + horizon],
                y_context=y_rollout[: initial_context_size + horizon],
                x_query=x_query,
                n_classes=n_classes,
            )
            distances.append(probability_distance(p_h, p0, distance_name=distance_name))

        emd_per_query = torch.stack(distances, dim=0).mean(dim=0)
        batch_losses.append(emd_per_query.mean())

    return torch.stack(batch_losses).mean()


def classification_self_consistency_loss(
    *,
    model: nn.Module,
    x_rollout: Tensor,
    y_rollout: Tensor,
    x_query: Tensor,
    initial_context_size: int,
    n_classes: int,
    eps: float = 1e-12,
) -> Tensor:
    """Differentiable SC loss on an already-fixed rollout trajectory."""

    teacher = query_probabilities_from_fixed_context(
        model,
        x_context=x_rollout[:initial_context_size],
        y_context=y_rollout[:initial_context_size],
        x_query=x_query,
        n_classes=n_classes,
    ).detach()

    prefix_losses: list[Tensor] = []
    for horizon in range(1, x_rollout.shape[0] - initial_context_size + 1):
        student = query_probabilities_from_fixed_context(
            model,
            x_context=x_rollout[: initial_context_size + horizon],
            y_context=y_rollout[: initial_context_size + horizon],
            x_query=x_query,
            n_classes=n_classes,
        )
        prefix_losses.append(soft_cross_entropy(teacher, student, eps=eps))

    loss_per_query = torch.stack(prefix_losses, dim=0).mean(dim=0)
    return loss_per_query.mean()


def classification_self_consistency_rollout_objective(
    *,
    model: nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    idx_context: Tensor,
    idx_query: Tensor,
    n_classes: int,
    baseline_pred_rule_factory: Callable[[], Any],
    rollout_length: int,
    rollout_seed: int,
    eps: float = 1e-12,
) -> Tensor:
    """SC loss over frozen baseline rollout prefixes.

    Rollout generation is no-grad and fixed. The differentiable loss is only
    the soft CE between stop-gradient p_phi,0 and prefix beliefs p_phi,k.
    """

    validate_batched_indices(idx_context, idx_query)

    batch_losses: list[Tensor] = []
    for batch_idx in range(idx_query.shape[0]):
        context_indices = idx_context[batch_idx]
        x_query = gather_rows(x_all, idx_query[batch_idx])
        x_rollout, y_rollout = fixed_baseline_rollout_tensors(
            baseline_pred_rule_factory=baseline_pred_rule_factory,
            x_all=x_all,
            y_all=y_all,
            context_indices=context_indices,
            rollout_length=rollout_length,
            rollout_seed=rollout_seed + batch_idx,
        )

        batch_losses.append(
            classification_self_consistency_loss(
                model=model,
                x_rollout=x_rollout,
                y_rollout=y_rollout,
                x_query=x_query,
                initial_context_size=context_indices.shape[0],
                n_classes=n_classes,
                eps=eps,
            )
        )

    return torch.stack(batch_losses).mean()


def martingale_loss_fn(
    *,
    task_type: str,
    model: nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    idx_context: Tensor,
    idx_query: Tensor,
    n_classes: int | None = None,
    baseline_pred_rule_factory: Callable[[], Any] | None = None,
    rollout_length: int = 30,
    rollout_seed: int = 0,
    distance_name: str = "l1",
    sc_eps: float = 1e-12,
) -> Tensor:
    if task_type == "classification":
        if n_classes is None:
            raise ValueError("n_classes is required for classification loss.")
        if baseline_pred_rule_factory is None:
            raise ValueError(
                "baseline_pred_rule_factory is required for classification loss."
            )
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
        )
    if task_type == "regression":
        raise NotImplementedError("Regression martingale loss is not implemented yet.")
    raise ValueError(f"Unsupported task_type={task_type}.")


def martingale_step_fn(
    *,
    model: nn.Module,
    loss_fn: Callable[..., Tensor],
    idx_train: Tensor,
    idx: Tensor,
    state: Any,
    context: dict[str, Any],
) -> Tensor:
    return loss_fn(
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
        distance_name=context.get("distance_name", "l1"),
        sc_eps=context.get("sc_eps", 1e-12),
    )


def evaluate_global_emd(
    *,
    model: nn.Module,
    x_all: Tensor,
    y_all: Tensor,
    query_idx: np.ndarray,
    n_classes: int,
    baseline_pred_rule_factory: Callable[[], Any],
    rollout_length: int,
    rollout_seed: int = 0,
    include_query_in_context: bool = True,
    distance_name: str = "l1",
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
        baseline_pred_rule_factory=baseline_pred_rule_factory,
        rollout_length=rollout_length,
        rollout_seed=rollout_seed,
        distance_name=distance_name,
    )
    return float(loss.detach().cpu())
