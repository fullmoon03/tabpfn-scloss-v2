"""Backward-compatible martingale fine-tuning exports.

Implementation lives in `data.py` and `objectives.py` so dataset utilities,
rollout objectives, and the generic training loop stay separated.
"""

from fine_tuning.data import (
    EncodedClassificationData,
    ValTestQuerySplit,
    encode_classification_dataset,
    make_eval_indices,
    sample_disjoint_val_test_queries,
)
from fine_tuning.objectives import (
    baseline_rollout_numpy,
    classification_global_emd_loss,
    classification_self_consistency_loss,
    classification_self_consistency_rollout_objective,
    evaluate_global_emd,
    martingale_loss_fn,
    martingale_step_fn,
    probability_distance,
    query_probabilities_from_fixed_context,
    soft_cross_entropy,
)

__all__ = [
    "EncodedClassificationData",
    "ValTestQuerySplit",
    "baseline_rollout_numpy",
    "classification_global_emd_loss",
    "classification_self_consistency_loss",
    "classification_self_consistency_rollout_objective",
    "encode_classification_dataset",
    "evaluate_global_emd",
    "make_eval_indices",
    "martingale_loss_fn",
    "martingale_step_fn",
    "probability_distance",
    "query_probabilities_from_fixed_context",
    "sample_disjoint_val_test_queries",
    "soft_cross_entropy",
]
