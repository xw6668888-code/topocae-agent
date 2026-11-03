from __future__ import annotations
from typing import Any
def _dependencies():
    try:
        import torch
        import torch.nn.functional as functional
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except ImportError as error:
        raise RuntimeError(
            "MFTReNet-SI-D0 model dependencies are unavailable. Use the isolated "
            "MFT environment; this is not a model score."
        ) from error
    return torch, functional, Data, DataLoader
import math
import numpy as np

def _face_focal_loss(logits, labels, gamma: float = 2.0):
    torch, functional, _, _ = _dependencies()
    probability = functional.softmax(logits, dim=1)
    selected = probability.gather(1, labels.view(-1, 1)).clamp_min(1e-7)
    return (-torch.pow(1.0 - selected, gamma) * torch.log(selected)).mean()


def _positive_edges(instance_id):
    torch, _, _, _ = _dependencies()
    edges = []
    for instance in sorted(set(instance_id.detach().cpu().tolist()) - {0}):
        nodes = torch.nonzero(instance_id == instance, as_tuple=False).flatten()
        if len(nodes) >= 2:
            edges.append(torch.combinations(nodes, r=2).t().contiguous())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=instance_id.device)
    return torch.cat(edges, dim=1)


def _negative_edges(instance_id, requested: int):
    """Sample unordered true-negative pairs without replacement."""
    torch, _, _, _ = _dependencies()
    count = len(instance_id)
    if count < 2 or requested < 1:
        return torch.empty((2, 0), dtype=torch.long, device=instance_id.device)
    pairs = torch.combinations(
        torch.arange(count, device=instance_id.device), r=2
    ).t().contiguous()
    left = instance_id[pairs[0]]
    right = instance_id[pairs[1]]
    positive = (left == right) & (left > 0)
    candidates = pairs[:, ~positive]
    if candidates.shape[1] <= requested:
        return candidates
    order = torch.randperm(candidates.shape[1], device=instance_id.device)[:requested]
    return candidates[:, order]


def training_loss(output: dict[str, Any], batch) -> dict[str, Any]:
    torch, functional, _, _ = _dependencies()
    class_loss = _face_focal_loss(output["class_logits"], batch.y_class)
    part_losses = []
    for part_index in range(len(batch.ptr) - 1):
        start = int(batch.ptr[part_index])
        end = int(batch.ptr[part_index + 1])
        count = end - start
        instances = batch.instance_id[start:end]
        positive = _positive_edges(instances)
        if positive.shape[1] == 0:
            continue
        negative = _negative_edges(instances, positive.shape[1])
        z = output["segment_z"][start:end]
        positive_logits = (z[positive[0]] * z[positive[1]]).sum(dim=1)
        positive_loss = functional.softplus(-positive_logits).mean()
        if negative.shape[1]:
            negative_logits = (z[negative[0]] * z[negative[1]]).sum(dim=1)
            reconstruction = positive_loss + functional.softplus(negative_logits).mean()
        else:
            reconstruction = positive_loss
        mu = output["segment_mu"][start:end]
        logstd = output["segment_logstd"][start:end]
        kl = -0.5 * torch.mean(
            torch.sum(1.0 + 2.0 * logstd - mu.square() - logstd.exp().square(), dim=1)
        )
        part_losses.append(reconstruction + kl / count)
    if part_losses:
        instance_loss = torch.stack(part_losses).mean()
        total = torch.sqrt(class_loss.clamp_min(1e-12) * instance_loss.clamp_min(1e-12))
    else:
        instance_loss = class_loss.new_zeros(())
        total = class_loss
    return {
        "total": total,
        "class": class_loss,
        "instance": instance_loss,
        "parts_with_positive_pairs": len(part_losses),
    }
