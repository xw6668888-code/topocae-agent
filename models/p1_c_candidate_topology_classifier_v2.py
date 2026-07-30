from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.p1_b_candidate_topology_classifier import (
    CandidateModelDimensions,
    CandidateTopologyClassifier,
)


MODEL_SCHEMA_VERSION = "topocae_p1_c_candidate_boundary_context_mlp_1"


class BoundaryContextCandidateTopologyClassifier(CandidateTopologyClassifier):
    """P1-B classifier plus mean/max pooling of one-hop boundary-neighbor faces."""

    def __init__(
        self,
        dimensions: CandidateModelDimensions,
        macro_classes: list[str] | None = None,
    ) -> None:
        super().__init__(dimensions, macro_classes=macro_classes)
        face_dim = dimensions.face_embedding_dim
        edge_dim = dimensions.edge_embedding_dim
        engineered_dim = dimensions.engineered_feature_dim
        hidden = dimensions.hidden_dim
        context_dim = face_dim * 5 + edge_dim * 4 + engineered_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(
        self,
        face_embeddings: torch.Tensor,
        internal_edge_embeddings: torch.Tensor,
        boundary_edge_embeddings: torch.Tensor,
        engineered_features: torch.Tensor,
        boundary_context_face_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        attention = torch.softmax(self.face_attention(face_embeddings).squeeze(-1), dim=0)
        face_mean = face_embeddings.mean(dim=0)
        face_max = face_embeddings.max(dim=0).values
        face_attention = (attention.unsqueeze(-1) * face_embeddings).sum(dim=0)
        internal_mean, internal_max = _mean_max(
            internal_edge_embeddings, self.dimensions.edge_embedding_dim, face_embeddings
        )
        boundary_mean, boundary_max = _mean_max(
            boundary_edge_embeddings, self.dimensions.edge_embedding_dim, face_embeddings
        )
        context_mean, context_max = _mean_max(
            boundary_context_face_embeddings, self.dimensions.face_embedding_dim, face_embeddings
        )
        context = self.context_encoder(torch.cat([
            face_mean, face_max, face_attention,
            internal_mean, internal_max, boundary_mean, boundary_max,
            context_mean, context_max, engineered_features,
        ], dim=0))
        face_hidden = F.gelu(
            self.face_projection(face_embeddings) + self.face_context(context).unsqueeze(0)
        )
        return {
            "macro_logits": self.macro_head(context),
            "validity_logit": self.validity_head(context).squeeze(-1),
            "membership_logits": self.membership_head(face_hidden).squeeze(-1),
            "boundary_logits": self.boundary_head(face_hidden).squeeze(-1),
            "semantic_role_logits": self.semantic_role_head(face_hidden),
            "face_attention_weights": attention,
            "candidate_embedding": context,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "schema_version": MODEL_SCHEMA_VERSION,
            "boundary_context": "mean_max_one_hop_external_face_embeddings",
        }


def _mean_max(
    values: torch.Tensor,
    dimension: int,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[0] == 0:
        zero = reference.new_zeros(dimension)
        return zero, zero
    return values.mean(dim=0), values.max(dim=0).values


__all__ = [
    "BoundaryContextCandidateTopologyClassifier",
    "CandidateModelDimensions",
    "MODEL_SCHEMA_VERSION",
]
