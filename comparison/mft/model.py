"""MFTReNet-SI-D0 final model adaptation.

The graph and UV encoders are imported from the immutable author checkout.
This module replaces the broken CGC wiring in ``models/model_v2.py`` with the
advertised two-specific-plus-one-shared expert stack for each supervised D0
task.  D0 has no feature-relation labels, so the relation task is deliberately
absent and the method must not be described as a full MFTReNet reproduction.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"

MODEL_CONFIG: dict[str, int] = {
    "num_specific_experts": 2,
    "num_shared_experts": 1,
    "node_attr_dim": 14,
    "node_attr_emb": 32,
    "node_grid_dim": 7,
    "node_grid_emb": 32,
    "edge_attr_dim": 15,
    "edge_attr_emb": 32,
    "edge_grid_dim": 12,
    "edge_grid_emb": 32,
    "graph_encoder_layers": 3,
    "classify_hidden_dim": 64,
    "segment_hidden_dim": 16,
    "num_classes": 19,
}


def model_contract() -> dict[str, Any]:
    """Return the dependency-free architecture contract used by registration."""
    return {
        "name": "MFTReNet-SI-D0",
        "claim_boundary": (
            "author-code-based two-task D0 adaptation; not the full published "
            "three-task MFTReNet reproduction"
        ),
        "tasks": ["19-way face classification", "same-instance grouping"],
        "omitted_task": {
            "name": "feature topological relationship prediction",
            "reason": "D0 has no corresponding relation-type supervision",
        },
        "expert_contract": {
            "per_task_specific_experts": 2,
            "shared_experts": 1,
            "gate_outputs_per_task": 3,
            "supervised_tasks": 2,
            "unique_graph_embeddings": 7,
            "count_explanation": (
                "four task-specific experts, one shared expert, and two "
                "task-specific gate encoders"
            ),
        },
        "input": {
            "face_attributes": 14,
            "edge_attributes": 15,
            "face_uv_grid": [7, 10, 10],
            "edge_u_grid": [12, 10],
        },
        "configuration": dict(MODEL_CONFIG),
    }


def _author_components():
    """Import only unchanged upstream encoders and verify their provenance."""
    vendor = VENDOR_ROOT.resolve()
    if not vendor.is_dir():
        raise FileNotFoundError(f"Missing immutable MFTReNet checkout: {vendor}")
    existing = sys.modules.get("models")
    if existing is not None:
        origin = Path(getattr(existing, "__file__", "") or ".").resolve()
        if not origin.is_relative_to(vendor):
            raise RuntimeError(f"Top-level 'models' is already bound outside MFTReNet: {origin}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    import torch
    from torch import nn
    from torch_geometric import nn as tgnn
    from torch_geometric.nn import LayerNorm, MLP
    from models.encoders.graph_encoder import GraphEncoder_MultiAggr
    from models.encoders.uvgrid_encoder import UVNetCurveEncoder, UVNetSurfaceEncoder

    origins = [
        Path(sys.modules[GraphEncoder_MultiAggr.__module__].__file__).resolve(),
        Path(sys.modules[UVNetCurveEncoder.__module__].__file__).resolve(),
    ]
    if not all(path.is_relative_to(vendor) for path in origins):
        raise RuntimeError(f"Author encoder import escaped immutable checkout: {origins}")
    return torch, nn, tgnn, LayerNorm, MLP, GraphEncoder_MultiAggr, UVNetCurveEncoder, UVNetSurfaceEncoder


def build_model(configuration: dict[str, int] | None = None):
    """Build the patched author-code-based two-task model.

    Classes are intentionally local to keep importing this module possible in
    the extraction environment, where PyTorch Geometric need not be installed.
    Checkpoints contain state dictionaries, never pickled model instances.
    """
    cfg = dict(MODEL_CONFIG)
    if configuration:
        unknown = set(configuration) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
        cfg.update({key: int(value) for key, value in configuration.items()})
    if cfg["num_specific_experts"] != 2 or cfg["num_shared_experts"] != 1:
        raise ValueError("Registered MFTReNet-SI-D0 requires two specific and one shared expert")

    (
        torch, nn, tgnn, LayerNorm, MLP, GraphEncoder_MultiAggr,
        UVNetCurveEncoder, UVNetSurfaceEncoder,
    ) = _author_components()

    class AuthorGraphEmbedding(nn.Module):
        """The GraphEmb equations from upstream ``models/model_v2.py``."""

        def __init__(self):
            super().__init__()
            self.node_attr_encoder = MLP(
                in_channels=cfg["node_attr_dim"],
                hidden_channels=cfg["node_attr_emb"] * 2,
                out_channels=cfg["node_attr_emb"],
                num_layers=2,
                dropout=0.25,
                act=nn.Mish(),
                norm=LayerNorm(cfg["node_attr_emb"] * 2),
            )
            self.node_grid_encoder = UVNetSurfaceEncoder(
                cfg["node_grid_dim"], cfg["node_grid_emb"]
            )
            self.edge_attr_encoder = MLP(
                in_channels=cfg["edge_attr_dim"],
                hidden_channels=cfg["edge_attr_emb"] * 2,
                out_channels=cfg["edge_attr_emb"],
                num_layers=2,
                dropout=0.25,
                act=nn.Mish(),
                norm=LayerNorm(cfg["edge_attr_emb"] * 2),
            )
            self.edge_grid_encoder = UVNetCurveEncoder(
                cfg["edge_grid_dim"], cfg["edge_grid_emb"]
            )
            self.node_emb = cfg["node_attr_emb"] + cfg["node_grid_emb"]
            self.final_out_emb = 2 * self.node_emb
            aggregators = [
                tgnn.MeanAggregation(),
                tgnn.SumAggregation(),
                tgnn.MaxAggregation(),
                tgnn.SoftmaxAggregation(learn=True),
            ]
            self.graph_encoder = GraphEncoder_MultiAggr(
                self.node_emb,
                cfg["edge_attr_emb"] + cfg["edge_grid_emb"],
                num_layers=cfg["graph_encoder_layers"],
                aggr=aggregators,
            )

        def forward(self, graph):
            face_attr = graph["face_attr"]
            face_grid = graph["face_grid"]
            edge_attr = graph["edge_attr"]
            edge_grid = graph["edge_grid"]
            edge_index = graph["edge_index"]
            batch = graph["batch"]
            if face_attr.ndim != 2 or face_attr.shape[1] != cfg["node_attr_dim"]:
                raise ValueError(f"Expected Nx14 face attributes, got {tuple(face_attr.shape)}")
            if face_grid.ndim != 4 or tuple(face_grid.shape[1:]) != (7, 10, 10):
                raise ValueError(f"Expected Nx7x10x10 face grids, got {tuple(face_grid.shape)}")
            if edge_attr.ndim != 2 or edge_attr.shape[1] != cfg["edge_attr_dim"]:
                raise ValueError(f"Expected Ex15 edge attributes, got {tuple(edge_attr.shape)}")
            if edge_grid.ndim != 3 or tuple(edge_grid.shape[1:]) != (12, 10):
                raise ValueError(f"Expected Ex12x10 edge grids, got {tuple(edge_grid.shape)}")
            if edge_index.shape != (2, edge_attr.shape[0]):
                raise ValueError("Physical edge rows are not aligned with edge_index")
            node_features = torch.cat(
                [self.node_attr_encoder(face_attr), self.node_grid_encoder(face_grid)], dim=1
            )
            edge_features = torch.cat(
                [self.edge_attr_encoder(edge_attr), self.edge_grid_encoder(edge_grid)], dim=1
            )
            local, global_feature = self.graph_encoder(
                node_features, edge_index, edge_features, batch
            )
            return torch.cat([local, global_feature[batch]], dim=1)

    class Gate(nn.Module):
        def __init__(self, expert_count: int):
            super().__init__()
            self.encoder = AuthorGraphEmbedding()
            self.linear = nn.Linear(self.encoder.final_out_emb, expert_count)

        def forward(self, graph):
            return torch.softmax(self.linear(self.encoder(graph)), dim=1)

    class PatchedTwoTaskCGC(nn.Module):
        def __init__(self):
            super().__init__()
            specific = cfg["num_specific_experts"]
            shared = cfg["num_shared_experts"]
            self.shared_experts = nn.ModuleList(
                [AuthorGraphEmbedding() for _ in range(shared)]
            )
            self.class_experts = nn.ModuleList(
                [AuthorGraphEmbedding() for _ in range(specific)]
            )
            self.segment_experts = nn.ModuleList(
                [AuthorGraphEmbedding() for _ in range(specific)]
            )
            expert_count = specific + shared
            self.class_gate = Gate(expert_count)
            self.segment_gate = Gate(expert_count)
            self.output_dim = 2 * (cfg["node_attr_emb"] + cfg["node_grid_emb"])

        @staticmethod
        def _mix(task_outputs, shared_outputs, weights):
            expert_outputs = torch.stack([*task_outputs, *shared_outputs], dim=1)
            if expert_outputs.shape[:2] != weights.shape:
                raise RuntimeError(
                    "CGC expert/gate mismatch: "
                    f"experts={tuple(expert_outputs.shape)}, gate={tuple(weights.shape)}"
                )
            return torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1)

        def forward(self, graph):
            shared = [expert(graph) for expert in self.shared_experts]
            cls_outputs = [expert(graph) for expert in self.class_experts]
            seg_outputs = [expert(graph) for expert in self.segment_experts]
            cls_weights = self.class_gate(graph)
            seg_weights = self.segment_gate(graph)
            return (
                self._mix(cls_outputs, shared, cls_weights),
                self._mix(seg_outputs, shared, seg_weights),
                cls_weights,
                seg_weights,
            )

    class MFTReNetSID0(nn.Module):
        def __init__(self):
            super().__init__()
            self.cgc = PatchedTwoTaskCGC()
            out_dim = self.cgc.output_dim
            hidden = cfg["classify_hidden_dim"]
            self.classify = MLP(
                in_channels=out_dim,
                hidden_channels=hidden,
                out_channels=cfg["num_classes"],
                num_layers=3,
                dropout=0.25,
                act=nn.Mish(),
                norm=LayerNorm(hidden),
            )
            self.classify_segment_embedding = MLP(
                in_channels=hidden,
                hidden_channels=hidden * 2,
                out_channels=hidden,
                num_layers=2,
                dropout=0.25,
                act=nn.Mish(),
                norm=LayerNorm(hidden * 2),
            )
            segment_input = hidden + out_dim
            self.segment_mu = MLP(
                in_channels=segment_input,
                hidden_channels=segment_input * 2,
                out_channels=cfg["segment_hidden_dim"],
                num_layers=2,
                dropout=0.25,
                act=nn.Mish(),
                norm=LayerNorm(segment_input * 2),
            )
            self.segment_logstd = MLP(
                in_channels=segment_input,
                hidden_channels=segment_input * 2,
                out_channels=cfg["segment_hidden_dim"],
                num_layers=2,
                dropout=0.25,
                act=nn.Mish(),
                norm=LayerNorm(segment_input * 2),
            )

        def forward(self, graph):
            class_feature, segment_feature, class_gate, segment_gate = self.cgc(graph)
            class_logits, class_embedding = self.classify(
                class_feature, return_emb=True
            )
            segment_class = self.classify_segment_embedding(class_embedding)
            combined = torch.cat([segment_feature, segment_class], dim=1)
            mu = self.segment_mu(combined)
            logstd = self.segment_logstd(combined).clamp(max=10.0)
            z = mu + torch.randn_like(logstd) * torch.exp(logstd) if self.training else mu
            return {
                "class_logits": class_logits,
                "segment_z": z,
                "segment_mu": mu,
                "segment_logstd": logstd,
                "class_gate": class_gate,
                "segment_gate": segment_gate,
            }

    model = MFTReNetSID0()
    model.model_contract = model_contract()
    return model


def parameter_counts(model) -> dict[str, int]:
    return {
        "all": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
