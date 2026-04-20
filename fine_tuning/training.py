from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


def loss_fn(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Placeholder for user-defined loss logic.

    Expected output: a scalar torch.Tensor suitable for `loss.backward()`.
    """

    raise NotImplementedError("Provide your own loss_fn callable.")


def step_fn(
    *,
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    idx_train: torch.Tensor,
    idx: torch.Tensor,
    state: "TrainState",
    context: dict[str, Any],
) -> torch.Tensor:
    """Placeholder for one train step's forward/loss construction.

    Plug in custom logic here or pass a callable to `train_full_ft`.
    Expected output: a scalar torch.Tensor.
    """

    raise NotImplementedError("Provide your own step_fn callable.")


@dataclass
class FullFTConfig:
    """Full fine-tuning loop config.

    `n_steps=-1` means run until early stopping. If no `eval_fn` is supplied,
    use a finite `n_steps`.
    """

    train_size: int
    n_steps: int = -1
    epoch_size: int = 10
    batch_size: int = 1
    context_size: int | None = None
    query_size: int = 1024
    patience: int = 16
    optimizer: dict[str, Any] = field(
        default_factory=lambda: {"type": "AdamW", "lr": 3e-5, "weight_decay": 0.0}
    )
    gradient_clipping_norm: float | None = None
    n_lr_warmup_epochs: int | None = None
    amp: bool = False
    randperm: bool = False
    seed: int = 0
    output_dir: str | Path | None = None
    save_best: bool = True


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_score: float = -math.inf
    best_step: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalResult:
    """Validation/evaluation callback output.

    `score` is maximized and used for checkpoint selection and early stopping.
    """

    score: float
    metrics: dict[str, Any] = field(default_factory=dict)


class CandidateQueue:
    """Produces prediction target indices; remaining train rows become context."""

    def __init__(self, train_size: int, n_candidates: int, device: torch.device) -> None:
        if not 0 < n_candidates < train_size:
            raise ValueError("n_candidates must be in (0, train_size).")
        self._train_size = train_size
        self._n_candidates = n_candidates
        self._device = device
        self._queue = torch.tensor([], dtype=torch.long, device=device)

    def __next__(self) -> torch.Tensor:
        if len(self._queue) < self._n_candidates:
            self._queue = torch.cat(
                [self._queue, torch.randperm(self._train_size, device=self._device)]
            )
        out, self._queue = self._queue.split(
            [self._n_candidates, len(self._queue) - self._n_candidates]
        )
        return out


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = -math.inf
        self.bad_updates = 0

    def update(self, score: float) -> bool:
        if score > self.best:
            self.best = score
            self.bad_updates = 0
            return True
        self.bad_updates += 1
        return False

    def should_stop(self) -> bool:
        return self.bad_updates >= self.patience


def zero_weight_decay_condition(module: nn.Module, parameter_name: str) -> bool:
    return parameter_name.endswith("bias") or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
        ),
    )


def make_parameter_groups(module: nn.Module) -> list[dict[str, Any]]:
    """Return full-FT parameter groups with no freezing."""

    zero_wd_params = {
        parameter
        for child in module.modules()
        for parameter_name, parameter in child.named_parameters(recurse=False)
        if zero_weight_decay_condition(child, parameter_name)
    }
    default_params = [p for p in module.parameters() if p not in zero_wd_params]
    return [
        {"params": default_params},
        {"params": list(zero_wd_params), "weight_decay": 0.0},
    ]


def make_optimizer(*, params: Any, type: str = "AdamW", **kwargs: Any) -> torch.optim.Optimizer:
    optimizer_cls = getattr(torch.optim, type)
    return optimizer_cls(params=params, **kwargs)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
    }
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, path)


def train_full_ft(
    *,
    model: nn.Module,
    config: FullFTConfig,
    step_fn: Callable[..., torch.Tensor] = step_fn,
    loss_fn: Callable[..., torch.Tensor] = loss_fn,
    eval_fn: Callable[..., EvalResult] | None = None,
    context: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> TrainState:
    """Run full fine-tuning.

    `step_fn` receives model, loss_fn, idx_train, idx, state, and context.
    `eval_fn`, if supplied, receives model, state, and context and returns EvalResult.
    """

    if context is None:
        context = {}
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train()

    params = make_parameter_groups(model)
    optimizer_kwargs = dict(config.optimizer)
    optimizer_type = optimizer_kwargs.pop("type", "AdamW")
    optimizer = make_optimizer(params=params, type=optimizer_type, **optimizer_kwargs)

    if config.n_lr_warmup_epochs is not None:
        n_warmup_steps = max(1, config.n_lr_warmup_epochs * config.epoch_size)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=n_warmup_steps
        )
    else:
        lr_scheduler = None

    output_dir = Path(config.output_dir) if config.output_dir is not None else None
    state = TrainState()
    early_stopping = EarlyStopping(config.patience)
    if config.context_size is not None:
        if not 0 < config.context_size <= config.train_size:
            raise ValueError("context_size must be in (0, train_size].")
        if config.query_size > config.context_size:
            raise ValueError("query_size must be <= context_size.")
    else:
        if config.query_size >= config.train_size:
            raise ValueError("query_size must be < train_size when context_size is unset.")

    candidate_queue = (
        CandidateQueue(
            config.train_size,
            n_candidates=config.batch_size * config.query_size,
            device=device,
        )
        if config.context_size is None
        else None
    )
    context_candidate_queue = (
        CandidateQueue(
            config.train_size,
            n_candidates=config.batch_size * config.context_size,
            device=device,
        )
        if config.context_size is not None and config.context_size < config.train_size
        else None
    )
    amp_enabled = config.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()

    if config.n_steps == -1 and eval_fn is None:
        raise ValueError("n_steps=-1 requires eval_fn for early stopping.")

    while config.n_steps == -1 or state.step < config.n_steps:
        epoch_losses: list[float] = []
        model.train()

        for _ in range(config.epoch_size):
            if config.n_steps != -1 and state.step >= config.n_steps:
                break

            if config.context_size is not None:
                if config.context_size == config.train_size:
                    idx_train = torch.arange(config.train_size, device=device).expand(
                        config.batch_size, config.train_size
                    )
                elif config.randperm:
                    idx_train = torch.stack(
                        [
                            torch.randperm(config.train_size, device=device)[
                                : config.context_size
                            ]
                            for _ in range(config.batch_size)
                        ],
                        dim=0,
                    )
                else:
                    assert context_candidate_queue is not None
                    idx_train = next(context_candidate_queue).view(
                        config.batch_size, config.context_size
                    )

                query_positions = torch.stack(
                    [
                        torch.randperm(config.context_size, device=device)[
                            : config.query_size
                        ]
                        for _ in range(config.batch_size)
                    ],
                    dim=0,
                )
                idx = torch.gather(idx_train, dim=1, index=query_positions)
            else:
                if config.randperm:
                    idx = torch.randperm(config.train_size, device=device)[
                        : config.batch_size * config.query_size
                    ]
                else:
                    assert candidate_queue is not None
                    idx = next(candidate_queue)
                idx = idx.view(config.batch_size, -1)

                mask = idx.new_ones((config.batch_size, config.train_size), dtype=torch.bool)
                mask[torch.arange(config.batch_size, device=device).unsqueeze(-1), idx] = False
                idx_train = (
                    torch.arange(config.train_size, device=device)
                    .expand(config.batch_size, config.train_size)[mask]
                    .view(config.batch_size, -1)
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=torch.bfloat16 if amp_enabled else None,
            ):
                loss = step_fn(
                    model=model,
                    loss_fn=loss_fn,
                    idx_train=idx_train,
                    idx=idx,
                    state=state,
                    context=context,
                )
            loss.backward()

            if config.gradient_clipping_norm is not None:
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), config.gradient_clipping_norm
                )
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_losses.append(float(loss.detach().cpu()))
            state.step += 1

        result = None
        if eval_fn is not None:
            model.eval()
            with torch.inference_mode():
                result = eval_fn(model=model, state=state, context=context)
            improved = early_stopping.update(result.score)
            if improved:
                state.best_score = result.score
                state.best_step = state.step
                if output_dir is not None and config.save_best:
                    save_checkpoint(
                        output_dir / "best_checkpoint.pt",
                        model=model,
                        optimizer=optimizer,
                        state=state,
                        lr_scheduler=lr_scheduler,
                    )
        else:
            improved = False

        state.history.append(
            {
                "epoch": state.epoch,
                "step": state.step,
                "mean_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else None,
                "eval": None if result is None else result.metrics | {"score": result.score},
                "improved": improved,
            }
        )
        state.epoch += 1

        if eval_fn is not None and early_stopping.should_stop():
            break

    if output_dir is not None:
        save_checkpoint(
            output_dir / "last_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            state=state,
            lr_scheduler=lr_scheduler,
        )

    return state


class FullFineTuner:
    """Small OO wrapper around `train_full_ft`."""

    def __init__(
        self,
        model: nn.Module,
        config: FullFTConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    def fit(
        self,
        *,
        step_fn: Callable[..., torch.Tensor],
        loss_fn: Callable[..., torch.Tensor],
        eval_fn: Callable[..., EvalResult] | None = None,
        context: dict[str, Any] | None = None,
    ) -> TrainState:
        return train_full_ft(
            model=self.model,
            config=self.config,
            step_fn=step_fn,
            loss_fn=loss_fn,
            eval_fn=eval_fn,
            context=context,
            device=self.device,
        )
