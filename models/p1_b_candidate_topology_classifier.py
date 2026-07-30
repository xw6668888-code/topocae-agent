"""Small trainable candidate-level topology classifier for P1-B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from utils.p1_b_candidate_features import MACRO_CLASSES


MODEL_SCHEMA_VERSION = "topocae_p1_b_candidate_topology_mlp_1"


@dataclass(frozen=True)
class CandidateModelDimensions:
    face_embedding_dim: int
    edge_embedding_dim: int
    engineered_feature_dim: int
    hidden_dim: int = 128
    semantic_role_count: int = 8


class CandidateTopologyClassifier(nn.Module):
    """Pool a candidate subgraph and predict class, validity, and boundaries.

    The model is intentionally compact. It consumes face-GNN embeddings when
    available; Graph v2 vectors are the deterministic fallback. Candidate faces
    remain ragged, so membership/boundary logits are returned per original face.
    """

    def __init__(self, dimensions: CandidateModelDimensions, macro_classes: list[str] | None = None) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.macro_classes = list(macro_classes or MACRO_CLASSES)
        face_dim = dimensions.face_embedding_dim
        edge_dim = dimensions.edge_embedding_dim
        engineered_dim = dimensions.engineered_feature_dim
        hidden = dimensions.hidden_dim

        self.face_attention = nn.Linear(face_dim, 1)
        context_dim = face_dim * 3 + edge_dim * 4 + engineered_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.macro_head = nn.Linear(hidden, len(self.macro_classes))
        self.validity_head = nn.Linear(hidden, 1)
        self.face_context = nn.Linear(hidden, hidden)
        self.face_projection = nn.Linear(face_dim, hidden)
        self.membership_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Linear(hidden, 1)
        self.semantic_role_head = nn.Linear(hidden, dimensions.semantic_role_count)

    def forward(
        self,
        face_embeddings: torch.Tensor,
        internal_edge_embeddings: torch.Tensor,
        boundary_edge_embeddings: torch.Tensor,
        engineered_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _validate_rank(face_embeddings, 2, "face_embeddings")
        _validate_rank(internal_edge_embeddings, 2, "internal_edge_embeddings")
        _validate_rank(boundary_edge_embeddings, 2, "boundary_edge_embeddings")
        if engineered_features.ndim == 2 and engineered_features.shape[0] == 1:
            engineered_features = engineered_features[0]
        _validate_rank(engineered_features, 1, "engineered_features")
        if face_embeddings.shape[0] == 0:
            raise ValueError("candidate must contain at least one face embedding")
        if face_embeddings.shape[1] != self.dimensions.face_embedding_dim:
            raise ValueError("face embedding dimension does not match model metadata")
        if internal_edge_embeddings.shape[1] != self.dimensions.edge_embedding_dim:
            raise ValueError("internal edge embedding dimension does not match model metadata")
        if boundary_edge_embeddings.shape[1] != self.dimensions.edge_embedding_dim:
            raise ValueError("boundary edge embedding dimension does not match model metadata")
        if engineered_features.shape[0] != self.dimensions.engineered_feature_dim:
            raise ValueError("engineered feature dimension does not match model metadata")

        attention_weights = torch.softmax(self.face_attention(face_embeddings).squeeze(-1), dim=0)
        face_mean = face_embeddings.mean(dim=0)
        face_max = face_embeddings.max(dim=0).values
        face_attention = (attention_weights.unsqueeze(-1) * face_embeddings).sum(dim=0)
        internal_mean, internal_max = _mean_max_pool(
            internal_edge_embeddings, self.dimensions.edge_embedding_dim, face_embeddings
        )
        boundary_mean, boundary_max = _mean_max_pool(
            boundary_edge_embeddings, self.dimensions.edge_embedding_dim, face_embeddings
        )
        context_input = torch.cat(
            [
                face_mean,
                face_max,
                face_attention,
                internal_mean,
                internal_max,
                boundary_mean,
                boundary_max,
                engineered_features,
            ],
            dim=0,
        )
        context = self.context_encoder(context_input)
        face_hidden = F.gelu(self.face_projection(face_embeddings) + self.face_context(context).unsqueeze(0))
        return {
            "macro_logits": self.macro_head(context),
            "validity_logit": self.validity_head(context).squeeze(-1),
            "membership_logits": self.membership_head(face_hidden).squeeze(-1),
            "boundary_logits": self.boundary_head(face_hidden).squeeze(-1),
            "semantic_role_logits": self.semantic_role_head(face_hidden),
            "face_attention_weights": attention_weights,
            "candidate_embedding": context,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "dimensions": self.dimensions.__dict__,
            "macro_classes": list(self.macro_classes),
            "parameter_count": self.parameter_count(),
            "heads": [
                "candidate_macro_logits",
                "candidate_validity_score",
                "face_membership_refinement",
                "face_boundary_refinement",
                "semantic_face_role_logits",
            ],
            "pooling": ["mean_node", "max_node", "attention_node", "mean_max_internal_edge", "mean_max_boundary_edge"],
        }


def candidate_multitask_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    configured = {
        "macro": 1.0,
        "validity": 0.5,
        "membership": 0.5,
        "boundary": 0.5,
        "semantic_role": 0.25,
        **dict(weights or {}),
    }
    validity_target = targets["validity"].float().reshape(())
    losses: dict[str, torch.Tensor] = {
        "validity": F.binary_cross_entropy_with_logits(outputs["validity_logit"], validity_target),
    }
    if int(validity_target.item()) == 1:
        losses["macro"] = F.cross_entropy(outputs["macro_logits"].unsqueeze(0), targets["macro_class"].long().reshape(1))
    else:
        losses["macro"] = outputs["macro_logits"].sum() * 0.0
    if "membership" in targets:
        losses["membership"] = F.binary_cross_entropy_with_logits(
            outputs["membership_logits"], targets["membership"].float()
        )
    else:
        losses["membership"] = outputs["membership_logits"].sum() * 0.0
    if "boundary" in targets:
        losses["boundary"] = F.binary_cross_entropy_with_logits(
            outputs["boundary_logits"], targets["boundary"].float()
        )
    else:
        losses["boundary"] = outputs["boundary_logits"].sum() * 0.0
    if "semantic_role" in targets:
        losses["semantic_role"] = F.cross_entropy(outputs["semantic_role_logits"], targets["semantic_role"].long())
    else:
        losses["semantic_role"] = outputs["semantic_role_logits"].sum() * 0.0
    total = sum(configured[name] * loss for name, loss in losses.items())
    return {"total": total, **losses}


def predict_candidate_record(
    model: CandidateTopologyClassifier,
    tensor_record: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    face_embeddings = torch.tensor(tensor_record["face_embeddings"], dtype=torch.float32, device=device)
    edge_dim = model.dimensions.edge_embedding_dim
    internal = _tensor_or_empty(tensor_record.get("internal_edge_embeddings", []), edge_dim, device)
    boundary = _tensor_or_empty(tensor_record.get("boundary_edge_embeddings", []), edge_dim, device)
    engineered = torch.tensor(tensor_record["engineered_features"], dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = model(face_embeddings, internal, boundary, engineered)
        macro_probabilities = torch.softmax(outputs["macro_logits"], dim=0)
        validity = torch.sigmoid(outputs["validity_logit"])
        membership = torch.sigmoid(outputs["membership_logits"])
        boundary_probabilities = torch.sigmoid(outputs["boundary_logits"])
        semantic_roles = torch.softmax(outputs["semantic_role_logits"], dim=1)
    return {
        "candidate_id": tensor_record.get("candidate_id"),
        "face_indices": list(tensor_record.get("face_indices", [])),
        "macro_classes": list(model.macro_classes),
        "macro_logits": outputs["macro_logits"].cpu().tolist(),
        "macro_probabilities": macro_probabilities.cpu().tolist(),
        "validity_score": float(validity.cpu()),
        "membership_probabilities": membership.cpu().tolist(),
        "boundary_probabilities": boundary_probabilities.cpu().tolist(),
        "semantic_role_probabilities": semantic_roles.cpu().tolist(),
        "face_attention_weights": outputs["face_attention_weights"].cpu().tolist(),
        "model_schema_version": MODEL_SCHEMA_VERSION,
    }


def _mean_max_pool(
    values: torch.Tensor,
    dimension: int,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[0] == 0:
        zero = reference.new_zeros((dimension,))
        return zero, zero
    return values.mean(dim=0), values.max(dim=0).values


def _tensor_or_empty(values: Any, dimension: int, device: str | torch.device) -> torch.Tensor:
    if values:
        return torch.tensor(values, dtype=torch.float32, device=device)
    return torch.empty((0, dimension), dtype=torch.float32, device=device)


def _validate_rank(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.ndim != expected:
        raise ValueError(f"{name} must have rank {expected}, got {tensor.ndim}")


__all__ = [
    "CandidateModelDimensions",
    "CandidateTopologyClassifier",
    "MODEL_SCHEMA_VERSION",
    "candidate_multitask_loss",
    "predict_candidate_record",
]
