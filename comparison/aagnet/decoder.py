import json
from pathlib import Path
import numpy as np
CLASSES = json.loads((Path(__file__).parent / "classes.json").read_text())

def decode(semantic, instance_probability, threshold=0.5, face_mapping=None):
    """Author row-greedy proposals and summed-logit voting; stock remapped to 0."""
    adj = instance_probability > threshold
    used = np.zeros(len(adj), dtype=bool)
    proposals = []
    for i, row in enumerate(adj):
        if used[i] or not row.any():
            continue
        faces = np.flatnonzero(row & ~used).tolist()
        if not faces:
            continue
        used[faces] = True
        c = int(semantic[faces].sum(axis=0).argmax())
        if c == 0:
            continue
        output_faces = faces if face_mapping is None else [int(face_mapping[f]) for f in faces]
        proposals.append({'class': CLASSES[c], 'faces': sorted(output_faces)})
    return sorted(proposals, key=lambda p: (p['class'], p['faces']))
