from __future__ import annotations
from pathlib import Path
from types import MethodType
import json
import os
os.environ.setdefault('DGLBACKEND', 'pytorch')
import numpy as np
import torch
from .vendor.models.inst_segmentors import AAGNetSegmentor


class TensorGraph:
    """Tensor storage and ordered incoming mailboxes for the author's PNA ops."""

    def __init__(self, inputs, device='cpu'):
        self.sizes = [len(x['node']) for x in inputs]
        offsets = np.cumsum([0] + self.sizes[:-1])
        self.ndata = {k: torch.as_tensor(np.concatenate([x[name] for x in inputs]), device=device)
                      for k, name in [('x', 'node'), ('grid', 'grid')]}
        self.edata = {'x': torch.as_tensor(np.concatenate([x['edge'] for x in inputs]), device=device)}
        edges = np.concatenate([x['edge_index'] + offset for x, offset in zip(inputs, offsets)], axis=1)
        self.src, self.dst = torch.as_tensor(edges, dtype=torch.long, device=device)
        n = sum(self.sizes)
        mailboxes = [[] for _ in range(n)]
        for e, dst in enumerate(edges[1]):
            mailboxes[int(dst)].append(e)
        width = max(1, max(map(len, mailboxes)))
        incoming = np.full((n, width), len(edges[0]), dtype=np.int64)
        for i, mailbox in enumerate(mailboxes):
            incoming[i, :len(mailbox)] = mailbox
        self.incoming = torch.as_tensor(incoming, device=device)
        self.incoming_valid = self.incoming != len(edges[0])
        self.has_incoming = self.incoming_valid.any(dim=1, keepdim=True)
        self.batch_index = torch.repeat_interleave(torch.arange(len(inputs), device=device),
                                                  torch.tensor(self.sizes, device=device))
        self.counts = torch.tensor(self.sizes, dtype=torch.long, device=device)
        self.snorm = self.counts[self.batch_index].to(torch.float32).rsqrt().unsqueeze(1)

    def batch_num_nodes(self):
        return self.counts

    def to(self, device):
        self.ndata = {k: v.to(device) for k, v in self.ndata.items()}
        self.edata = {k: v.to(device) for k, v in self.edata.items()}
        for key in ['src', 'dst', 'incoming', 'incoming_valid', 'has_incoming', 'batch_index', 'counts', 'snorm']:
            setattr(self, key, getattr(self, key).to(device))
        return self


def pna_forward(self, graph, node_feat, edge_feat=None):
    messages = self.M(torch.cat([node_feat[graph.src], node_feat[graph.dst], edge_feat], dim=-1))
    padded = torch.cat([messages, messages.new_zeros(1, messages.shape[1])], dim=0)
    mailbox = padded[graph.incoming]
    total = mailbox.sum(dim=1)
    # max(dim=1) retains the author's first-maximum gradient convention.
    maximum = mailbox.masked_fill(~graph.incoming_valid.unsqueeze(-1), -torch.inf).max(dim=1).values
    maximum = torch.where(graph.has_incoming, maximum, torch.zeros_like(maximum))
    h = self.U(torch.cat([node_feat, total, maximum], dim=-1))
    return self.norm_drop(h * graph.snorm.to(h))


def pool_forward(self, graph, feat):
    pooled = feat.new_zeros(len(graph.sizes), feat.shape[-1])
    pooled.index_add_(0, graph.batch_index, feat)
    return pooled / graph.counts.to(feat).unsqueeze(1)


def build_model(backend='torch'):
    config = json.loads((Path(__file__).parent / 'architecture.json').read_text())
    assert not config['conv_on_edge']
    model = AAGNetSegmentor(**config)
    if backend == 'torch':
        for layer in model.graph_encoder.node_convs:
            layer.conv.forward = MethodType(pna_forward, layer.conv)
        model.graph_encoder.pool.forward = MethodType(pool_forward, model.graph_encoder.pool)
    elif backend != 'dgl':
        raise ValueError(backend)
    # There is no native MFInstSeg bottom annotation in this dataset.
    for parameter in model.bottom_head.parameters():
        parameter.requires_grad_(False)
    return model
