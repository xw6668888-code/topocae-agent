"""Inference on geometry-only author native arrays, using relocated frozen assets."""
import json
from pathlib import Path
import numpy as np
import torch
from .model import TensorGraph, build_model
from .normalization import normalize
from .decoder import decode

BASE = Path(__file__).parent

def predict(native, seed=20260719):
    cfg = json.loads((BASE/'config.json').read_text())
    row = cfg['seeds'][str(seed)]
    model = build_model()
    checkpoint = torch.load(BASE/row['original_weights'], map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    stats = json.loads((BASE/'normalization.json').read_text())
    graph = TensorGraph([normalize(native, stats)])
    with torch.inference_mode():
        semantic, instances, _ = model(graph)
        semantic = semantic.cpu().numpy()
        probability = torch.sigmoid(instances[0]).cpu().numpy()
    threshold = cfg['main_threshold']
    mapping = np.asarray(native['native_to_step'])
    if sorted(mapping.tolist()) != list(range(len(semantic))):
        raise ValueError('Native-to-STEP mapping must be bijective')
    return decode(semantic, probability, threshold, mapping)
