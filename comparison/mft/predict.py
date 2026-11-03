"""Frozen two-task MFT inference from author geometry arrays; no annotations."""
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from .model import build_model
from .native_contract import validate_native
from .decoder import decode_instances

BASE = Path(__file__).parent

def predict(native, seed=20260719):
    cfg = json.loads((BASE/'config.json').read_text())['seeds'][str(seed)]
    norm = json.loads((BASE/'normalization.json').read_text())
    n = len(native['face_attr'])
    mapping = validate_native(native,n)
    model = build_model()
    cp = torch.load(BASE/cfg['weights'], map_location='cpu', weights_only=False)
    model.load_state_dict(cp['model'],strict=True)
    model.eval()
    arrays = dict(native)
    for name in ('face_attr','edge_attr'):
        arrays[name] = (arrays[name]-np.asarray(norm[name]['mean'],dtype=np.float32))/np.asarray(norm[name]['std'],dtype=np.float32)
    graph = Data(num_nodes=n, **{k:torch.as_tensor(v,dtype=torch.long if k=='edge_index' else torch.float32)
                                for k,v in arrays.items() if k!='native_to_step'})
    with torch.inference_mode():
        out = model(Batch.from_data_list([graph]))
        logits = out['class_logits'].float().cpu().numpy()
        pairs = (out['segment_z']@out['segment_z'].t()).float().cpu().numpy()
    return decode_instances(logits,pairs,threshold=cfg['threshold'],native_to_step=mapping)
