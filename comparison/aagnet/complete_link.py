from .decoder import CLASSES
import numpy as np

def decode_complete_link(semantic, probability, threshold, face_mapping, *, tie_tolerance=1e-7):
    """Merge same-predicted-class clusters only when *all* cross links agree.

    A conflicting exact score tie is left unmerged, never broken by face order.
    The symmetric minimum of the two directed predictions prevents one
    high-confidence direction alone from imposing an instance link.
    """
    logits = np.asarray(semantic)
    relation = np.asarray(probability)
    mapping = np.asarray(face_mapping)
    n = len(logits)
    if logits.ndim != 2 or logits.shape[1] != 19 or relation.shape != (n, n):
        raise ValueError("AAGNet logits/relation dimensions changed")
    if sorted(mapping.tolist()) != list(range(n)):
        raise ValueError("Native-to-STEP mapping is not bijective")
    if not np.isfinite(logits).all() or not np.isfinite(relation).all():
        raise ValueError("AAGNet output contains nonfinite values")
    if not 0 < threshold < 1:
        raise ValueError("Invalid relation threshold")
    classes = logits.argmax(axis=1)
    agreement = np.minimum(relation, relation.T)
    clusters = [frozenset((i,)) for i in range(n) if classes[i] != 0]
    ambiguous = 0
    blocked_faces = set()
    while len(clusters) > 1:
        pairs = []
        for a, left in enumerate(clusters):
            if left.intersection(blocked_faces):
                continue
            for b in range(a + 1, len(clusters)):
                right = clusters[b]
                if right.intersection(blocked_faces):
                    continue
                if classes[next(iter(left))] != classes[next(iter(right))]:
                    continue
                support = min(float(agreement[i, j]) for i in left for j in right)
                if support > threshold:
                    pairs.append((support, a, b))
        if not pairs:
            break
        best = max(row[0] for row in pairs)
        ties = [(a, b) for score, a, b in pairs if best - score <= tie_tolerance]
        counts = {index: sum(index in pair for pair in ties)
                  for pair in ties for index in pair}
        conflicting = {index for index, count in counts.items() if count > 1}
        if conflicting:
            peers = {index for pair in ties if any(index in pair for index in conflicting)
                     for index in pair}
            blocked_faces.update(face for index in peers for face in clusters[index])
            ambiguous += len([pair for pair in ties if set(pair).intersection(conflicting)])
            continue
        # Disjoint tied merges commute; no face-index or input-row order enters
        # the accepted partition. Competing ties are frozen above.
        a, b = ties[0]
        clusters = [item for k, item in enumerate(clusters) if k not in (a, b)] + [clusters[a] | clusters[b]]
    objects = [{"macro_feature": CLASSES[int(classes[next(iter(group))])],
                "face_indices": sorted(int(mapping[i]) for i in group)} for group in clusters]
    objects.sort(key=lambda row: (row["macro_feature"], row["face_indices"]))
    macro = [None] * n
    for i, label in enumerate(classes):
        macro[int(mapping[i])] = CLASSES[int(label)]
    return objects, macro, {"ambiguous_tied_merges": ambiguous}
