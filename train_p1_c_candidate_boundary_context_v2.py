from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

import train_p1_c_candidate_classifier_v2 as balanced
from models.p1_c_candidate_topology_classifier_v2 import (
    BoundaryContextCandidateTopologyClassifier,
)
from utils.p1_c_candidate_iteration_paths import CandidateIterationRunRoot


protocol = balanced.protocol
protocol.RUN_ROOT = CandidateIterationRunRoot()
protocol.CandidateTopologyClassifier = BoundaryContextCandidateTopologyClassifier


def _context_collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    batch = balanced._balanced_collate(records)
    dimension = int(records[0]["face_embeddings"].shape[1])
    maximum = max(max(int(record["boundary_context_face_embeddings"].shape[0]) for record in records), 1)
    values = torch.zeros((len(records), maximum, dimension), dtype=torch.float32)
    mask = torch.zeros((len(records), maximum), dtype=torch.bool)
    for row, record in enumerate(records):
        context = record["boundary_context_face_embeddings"].float()
        count = int(context.shape[0])
        if count:
            values[row, :count] = context
            mask[row, :count] = True
    batch["boundary_context_faces"] = values
    batch["boundary_context_mask"] = mask
    return batch


def _context_batch_forward(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    faces, face_mask = batch["faces"], batch["face_mask"]
    scores = model.face_attention(faces).squeeze(-1).masked_fill(~face_mask, -1e9)
    attention = torch.softmax(scores, dim=1)
    face_count = face_mask.sum(dim=1, keepdim=True).clamp_min(1)
    face_mean = (faces * face_mask.unsqueeze(-1)).sum(dim=1) / face_count
    face_max = faces.masked_fill(~face_mask.unsqueeze(-1), -1e9).max(dim=1).values
    face_attention = (attention.unsqueeze(-1) * faces).sum(dim=1)
    internal_mean, internal_max = protocol._masked_mean_max(batch["internal"], batch["internal_mask"])
    boundary_mean, boundary_max = protocol._masked_mean_max(
        batch["boundary_edges"], batch["boundary_edge_mask"]
    )
    context_mean, context_max = protocol._masked_mean_max(
        batch["boundary_context_faces"], batch["boundary_context_mask"]
    )
    context = model.context_encoder(torch.cat([
        face_mean, face_max, face_attention,
        internal_mean, internal_max, boundary_mean, boundary_max,
        context_mean, context_max, batch["engineered"],
    ], dim=1))
    face_hidden = F.gelu(
        model.face_projection(faces) + model.face_context(context).unsqueeze(1)
    )
    return {
        "macro_logits": model.macro_head(context),
        "validity_logits": model.validity_head(context).squeeze(-1),
        "membership_logits": model.membership_head(face_hidden).squeeze(-1),
        "boundary_logits": model.boundary_head(face_hidden).squeeze(-1),
        "semantic_logits": model.semantic_role_head(face_hidden),
        "attention": attention,
    }


protocol._collate = _context_collate
protocol._batch_forward = _context_batch_forward


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
