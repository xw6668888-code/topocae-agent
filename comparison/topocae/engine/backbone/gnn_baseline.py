from __future__ import annotations

import torch
from torch import nn


class MeanGraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            neighbor_mean = torch.zeros_like(x)
        else:
            src, dst = edge_index[0], edge_index[1]
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, dst, x[src])

            degrees = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            degrees.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            neighbor_mean = aggregated / degrees.clamp_min(1.0).unsqueeze(-1)

        out = self.linear(torch.cat([x, neighbor_mean], dim=-1))
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class EdgeAwareMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_feature_dim: int, dropout: float = 0.0):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            neighbor_mean = torch.zeros_like(x)
        else:
            src, dst = edge_index[0], edge_index[1]
            messages = self.message_mlp(torch.cat([x[src], edge_features], dim=-1))
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, dst, messages)

            degrees = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            degrees.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            neighbor_mean = aggregated / degrees.clamp_min(1.0).unsqueeze(-1)

        return self.update(torch.cat([x, neighbor_mean], dim=-1))


class MultiHeadGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_primitive_surfaces: int,
        num_macro_classes: int,
        num_semantic_classes: int,
        derived_target_dim: int,
        idealization_target_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        primitive_embedding_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.primitive_embedding = nn.Embedding(num_primitive_surfaces, primitive_embedding_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim + primitive_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            MeanGraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.macro_head = nn.Linear(hidden_dim, num_macro_classes)
        self.semantic_head = nn.Linear(hidden_dim, num_semantic_classes)
        self.derived_head = nn.Linear(hidden_dim, derived_target_dim) if derived_target_dim > 0 else None
        self.idealization_head = nn.Linear(hidden_dim, idealization_target_dim) if idealization_target_dim > 0 else None

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        primitive_surface_ids: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        primitive_embedding = self.primitive_embedding(primitive_surface_ids)
        x = self.input_projection(torch.cat([node_features, primitive_embedding], dim=-1))
        for layer in self.layers:
            x = x + layer(x, edge_index)

        outputs = {
            "macro_logits": self.macro_head(x),
            "semantic_logits": self.semantic_head(x),
            "node_embeddings": x,
        }
        if self.derived_head is not None:
            outputs["derived_logits"] = self.derived_head(x)
        if self.idealization_head is not None:
            outputs["idealization_logits"] = self.idealization_head(x)
        return outputs


class EdgeAwareMultiHeadGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_feature_dim: int,
        num_primitive_surfaces: int,
        num_macro_classes: int,
        num_semantic_classes: int,
        derived_target_dim: int,
        idealization_target_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        primitive_embedding_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if edge_feature_dim < 1:
            raise ValueError("edge_feature_dim must be at least 1")

        self.edge_feature_dim = edge_feature_dim
        self.primitive_embedding = nn.Embedding(num_primitive_surfaces, primitive_embedding_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim + primitive_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            EdgeAwareMessagePassingLayer(hidden_dim, edge_feature_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.macro_head = nn.Linear(hidden_dim, num_macro_classes)
        self.semantic_head = nn.Linear(hidden_dim, num_semantic_classes)
        self.derived_head = nn.Linear(hidden_dim, derived_target_dim) if derived_target_dim > 0 else None
        self.idealization_head = nn.Linear(hidden_dim, idealization_target_dim) if idealization_target_dim > 0 else None
        self.edge_same_instance_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        primitive_surface_ids: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if edge_features is None:
            edge_features = torch.zeros(
                (edge_index.size(1), self.edge_feature_dim),
                device=node_features.device,
                dtype=node_features.dtype,
            )

        primitive_embedding = self.primitive_embedding(primitive_surface_ids)
        x = self.input_projection(torch.cat([node_features, primitive_embedding], dim=-1))
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_features)

        outputs = {
            "macro_logits": self.macro_head(x),
            "semantic_logits": self.semantic_head(x),
            "node_embeddings": x,
        }
        if self.derived_head is not None:
            outputs["derived_logits"] = self.derived_head(x)
        if self.idealization_head is not None:
            outputs["idealization_logits"] = self.idealization_head(x)

        if edge_index.numel() == 0:
            outputs["edge_same_instance_logits"] = torch.zeros(
                0,
                device=node_features.device,
                dtype=node_features.dtype,
            )
        else:
            src, dst = edge_index[0], edge_index[1]
            edge_embedding = self.edge_projection(edge_features)
            edge_pair_features = torch.cat([x[src], x[dst], edge_embedding], dim=-1)
            outputs["edge_same_instance_logits"] = self.edge_same_instance_head(edge_pair_features).squeeze(-1)
        return outputs
