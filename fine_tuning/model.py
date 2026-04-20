from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from fine_tuning.tabpfn_model.layer import PerFeatureEncoderLayer


class LayerStack(nn.Module):
    """A minimal LayerStack compatible with TabPFN's per-feature encoder layers."""

    def __init__(
        self,
        *,
        layer_creator,
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(self, x: Tensor, *, half_layers: bool = False, **kwargs: Any) -> Tensor:
        if half_layers:
            assert self.min_num_layers_layer_dropout == self.num_layers
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)
            else:
                x = layer(x, **kwargs)
        return x


class LinearEmbeddings(nn.Module):
    """Continuous feature embeddings with shape (*, n_features) -> (*, n_features, d)."""

    def __init__(self, n_features: int, d_embedding: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is None:
            return torch.mul(self.weight, x[..., None])
        return torch.addcmul(self.bias, self.weight, x[..., None])


class CategoricalEmbeddings1d(nn.Module):
    """Categorical feature embeddings with shape (*, n_cat) -> (*, n_cat, d)."""

    def __init__(self, cardinalities: list[int], d_embedding: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, d_embedding) for cardinality in cardinalities]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            bound = embedding.weight.shape[-1] ** -0.5
            nn.init.uniform_(embedding.weight, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([embedding(x[..., i]) for i, embedding in enumerate(self.embeddings)], dim=-2)


@dataclass
class TabPFN2Config:
    """Portable TabPFN2 wrapper config.

    `checkpoint_path` must point to a TabPFN v2 classifier/regressor checkpoint.
    The default architecture values match the checkpoints used by this project.
    """

    checkpoint_path: str | Path
    is_regression: bool
    n_num_features: int
    cat_cardinalities: list[int] = field(default_factory=list)
    n_classes: int | None = None
    n_bin_features: int = 0
    tabpfn_config: dict[str, int] = field(
        default_factory=lambda: {
            "nhead": 6,
            "emsize": 192,
            "nlayers": 12,
            "nhid_factor": 4,
        }
    )
    untie_value_embeddings: bool = False
    untie_pos_embeddings: bool = False
    affine_layer_norm: bool = False


class TabPFN2(nn.Module):
    """TabPFN v2 model wrapper used by the full fine-tuning pipeline."""

    def __init__(self, config: TabPFN2Config) -> None:
        super().__init__()
        if config.n_bin_features:
            raise ValueError(
                "This portable wrapper expects binary features to be provided either "
                "as numeric features or as categorical features with cardinality 2."
            )
        if not (config.n_num_features or config.cat_cardinalities):
            raise ValueError("At least one numeric or categorical feature is required.")

        self.is_regression = config.is_regression
        self.cat_cardinalities = config.cat_cardinalities
        self.untie_value_embeddings = config.untie_value_embeddings
        self.untie_pos_embeddings = config.untie_pos_embeddings

        checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]

        def extract_state_dict(prefix: str) -> dict[str, Tensor]:
            return {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }

        tabpfn_config = config.tabpfn_config
        emsize = tabpfn_config["emsize"]

        pos_embs = nn.Linear(emsize // 4, emsize)
        pos_embs.load_state_dict(extract_state_dict("feature_positional_embedding_embeddings."))
        self.pos_embs = None if config.untie_pos_embeddings else pos_embs

        if config.n_num_features > 0 and config.untie_value_embeddings:
            self.m_num = LinearEmbeddings(
                config.n_num_features,
                emsize,
                bias=config.untie_pos_embeddings,
            )
            weight_init = (
                state_dict["encoder.5.layer.weight"][:, 0]
                .unsqueeze(0)
                .repeat(config.n_num_features, 1)
            )
            self.m_num.weight.data = weight_init
            if config.untie_pos_embeddings:
                self.m_num.bias.data = pos_embs(  # type: ignore[union-attr]
                    torch.randn(config.n_num_features, emsize // 4)
                )
        elif config.n_num_features > 0 and not config.untie_pos_embeddings:
            self.m_num = nn.Linear(1, emsize, bias=False)
            self.m_num.weight.data = state_dict["encoder.5.layer.weight"][:, 0].unsqueeze(1)
        elif config.n_num_features > 0:
            raise ValueError("untie_pos_embeddings=True requires untie_value_embeddings=True.")
        else:
            self.m_num = None

        if config.cat_cardinalities and config.untie_value_embeddings:
            self.m_cat = CategoricalEmbeddings1d(config.cat_cardinalities, emsize)
            cat_pos_embs = (
                pos_embs(torch.randn(len(config.cat_cardinalities), emsize // 4))
                if config.untie_pos_embeddings
                else None
            )
            for i, cardinality in enumerate(config.cat_cardinalities):
                pos = cat_pos_embs[i].unsqueeze(0) if cat_pos_embs is not None else torch.zeros(1, emsize)
                value = (
                    state_dict["encoder.5.layer.weight"][:, 0].unsqueeze(0)
                    * torch.arange(cardinality + 1).unsqueeze(1)
                    / max(cardinality - 1, 1)
                )
                self.m_cat.embeddings[i].weight.data = pos + value
        elif config.cat_cardinalities and not config.untie_pos_embeddings:
            if config.n_num_features > 0:
                self.m_cat = self.m_num
            else:
                self.m_cat = nn.Linear(1, emsize, bias=False)
                self.m_cat.weight.data = state_dict["encoder.5.layer.weight"][:, 0].unsqueeze(1)
        elif config.cat_cardinalities:
            raise ValueError("untie_pos_embeddings=True requires untie_value_embeddings=True.")
        else:
            self.m_cat = None

        layer_key = "1" if config.is_regression else "2"
        self.y_embedding_weight = nn.Parameter(state_dict[f"y_encoder.{layer_key}.layer.weight"][:, 0])
        self.y_embedding_nan_ind = nn.Parameter(state_dict[f"y_encoder.{layer_key}.layer.weight"][:, 1])
        self.y_embedding_bias = nn.Parameter(state_dict[f"y_encoder.{layer_key}.layer.bias"])

        nhead = tabpfn_config["nhead"]
        nhid = emsize * tabpfn_config["nhid_factor"]
        nlayers = tabpfn_config["nlayers"]

        layer_creator = lambda: PerFeatureEncoderLayer(
            d_model=emsize,
            nhead=nhead,
            dim_feedforward=nhid,
            activation="gelu",
            zero_init=False,
            precomputed_kv=None,
            multiquery_item_attention_for_test_set=True,
            layer_norm_with_elementwise_affine=config.affine_layer_norm,
        )
        self.transformer_encoder = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers,
            recompute_each_layer=False,
            min_num_layers_layer_dropout=None,
        )
        self.transformer_encoder.load_state_dict(
            extract_state_dict("transformer_encoder."),
            strict=not config.affine_layer_norm,
        )

        # TabPFN v2 classifier checkpoint has a 10-logit standard decoder.
        # For binary tasks, slice logits in the user-defined step/eval function.
        n_outputs = 5000 if config.is_regression else 10
        self.decoder = nn.Sequential(
            nn.Linear(emsize, nhid),
            nn.GELU(),
            nn.Linear(nhid, n_outputs),
        )
        self.decoder.load_state_dict(extract_state_dict("decoder_dict.standard."))

    def forward(
        self,
        *,
        x_num: Tensor | None = None,
        x_cat: Tensor | None = None,
        y_train: Tensor,
    ) -> Tensor:
        batch_size = y_train.shape[0]
        train_size = y_train.shape[1]

        x_parts = []
        if x_num is not None:
            if self.untie_value_embeddings:
                x_parts.append(self.m_num(x_num) if self.m_num is not None else x_num)
            else:
                x_parts.append(self.m_num(x_num.unsqueeze(-1)) if self.m_num is not None else x_num)

        if x_cat is not None:
            if self.m_cat is None:
                raise ValueError("x_cat was provided, but the model has no categorical embeddings.")
            if not self.untie_value_embeddings:
                x_cat_max = x_cat[:, :train_size].max(dim=1, keepdim=True).values.clamp_min(1)
                x_cat = (x_cat / x_cat_max).unsqueeze(-1)
            x_parts.append(self.m_cat(x_cat))

        if not x_parts:
            raise ValueError("At least one of x_num or x_cat must be provided.")

        x_inp = torch.cat(x_parts, dim=2)
        total_size = x_inp.shape[1]

        y_train_float = y_train.float()
        y_mult = y_train_float.mean(dim=1, keepdim=True)
        if not self.is_regression:
            y_mult = torch.round(y_mult)
        y_test = x_inp.new_ones(batch_size, total_size - train_size) * y_mult
        nan_ind = x_inp.new_zeros(batch_size, total_size)
        nan_ind[:, train_size:] = -2.0
        y_emb = (
            torch.cat([y_train_float, y_test], dim=1).view(batch_size, -1, 1, 1)
            * self.y_embedding_weight.view(1, 1, 1, -1)
            + nan_ind.view(batch_size, -1, 1, 1)
            * self.y_embedding_nan_ind.view(1, 1, 1, -1)
            + self.y_embedding_bias.view(1, 1, 1, -1)
        )

        if self.pos_embs is not None:
            _, _, n_features, d_emb = x_inp.shape
            x_inp = x_inp + self.pos_embs(
                torch.randn(n_features, d_emb // 4, device=x_inp.device)
            )[None, None]

        x_inp = torch.cat([x_inp, y_emb], dim=2)
        encoder_out = self.transformer_encoder(
            x_inp,
            half_layers=False,
            cache_trainset_representation=False,
            single_eval_pos=train_size,
        )
        return self.decoder(encoder_out[:, train_size:, -1])
