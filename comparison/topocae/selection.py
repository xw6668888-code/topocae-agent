"""Whole-object selection with exact ambiguity abstention and bounded fallback.

The objective is maximum total *positive* complete-object score, with no face
coverage constraint.  Face indices determine conflicts, never decision ties.
"""

from __future__ import annotations

import math


TRANSITIONS = frozenset(("fillet", "chamfer"))


def objects_conflict(left, right):
    """Recorded P1 ontology: transition-to-main attachment may share faces.

    An exact duplicate face set cannot represent two distinct feature objects.
    Main objects, including a pocket and an embedded thin wall, may coexist
    when disjoint, regardless of physical adjacency or shared macro class.
    """
    a, b = set(left["face_indices"]), set(right["face_indices"])
    if not a.intersection(b):
        return False
    if a == b:
        return True
    ta, tb = left["macro_feature"] in TRANSITIONS, right["macro_feature"] in TRANSITIONS
    return ta == tb


def _candidate(row):
    faces = row["face_indices"]
    if not faces or len(set(faces)) != len(faces) or any(
        not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in faces
    ):
        raise ValueError("Candidates need nonempty distinct nonnegative integer faces")
    macro = row["macro_feature"]
    if not isinstance(macro, str) or not macro:
        raise ValueError("Candidate class must be nonempty text")
    score = row["score"]
    if isinstance(score, bool) or not math.isfinite(float(score)):
        raise ValueError("Candidate score must be finite")
    features = row.get("features", {})
    vector = features.get("vector", ()) if isinstance(features, dict) else ()
    if any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("Numeric evidence must be finite")
    return {"macro_feature": macro, "face_indices": tuple(sorted(faces)),
            "score": float(score), "evidence_key": tuple(float(value) for value in vector)}


def _deduplicate(candidates):
    unique = {}
    for row in candidates:
        candidate = _candidate(row)
        key = (candidate["macro_feature"], candidate["face_indices"])
        existing = unique.get(key)
        if existing is None or candidate["score"] > existing["score"]:
            unique[key] = candidate
        elif candidate["score"] == existing["score"] and candidate["evidence_key"] > existing["evidence_key"]:
            # Equivalent objects have the same output irrespective of their
            # metadata.  This choice affects only diagnostic evidence order.
            unique[key] = candidate
    return list(unique.values())


def _components(rows):
    n = len(rows)
    adjacency = [set() for _ in rows]
    for a in range(n):
        for b in range(a + 1, n):
            if objects_conflict(rows[a], rows[b]):
                adjacency[a].add(b)
                adjacency[b].add(a)
    pending = set(range(n))
    result = []
    while pending:
        frontier = [pending.pop()]
        part = set(frontier)
        while frontier:
            for neighbor in adjacency[frontier.pop()]:
                if neighbor in pending:
                    pending.remove(neighbor)
                    frontier.append(neighbor)
                    part.add(neighbor)
        result.append((list(part), adjacency))
    return result


def _exact(rows, adjacency, tolerance):
    count = len(rows)
    weights = [row["score"] for row in rows]
    neighbors = [sum(1 << j for j in adjacency[i]) for i in range(count)]
    best = -math.inf
    common = 0
    union = 0
    optima_seen = 0
    visits = 0

    def search(remaining, chosen, value):
        nonlocal best, common, union, optima_seen, visits
        visits += 1
        if not remaining:
            if value > best + tolerance:
                best, common, union, optima_seen = value, chosen, chosen, 1
            elif abs(value - best) <= tolerance:
                common &= chosen
                union |= chosen
                optima_seen += 1
            return
        upper = value + sum(weights[i] for i in range(count) if remaining & (1 << i))
        if upper < best - tolerance:
            return
        # Degree and evidence set traversal order, not the accepted answer.
        choice = max((i for i in range(count) if remaining & (1 << i)),
                     key=lambda i: (len(adjacency[i]), weights[i], rows[i]["macro_feature"], rows[i]["evidence_key"]))
        bit = 1 << choice
        search(remaining & ~bit & ~neighbors[choice], chosen | bit, value + weights[choice])
        search(remaining & ~bit, chosen, value)

    search((1 << count) - 1, 0, 0.0)
    return {i for i in range(count) if common & (1 << i)}, {
        "method": "exact", "objective": best,
        "selected_score": sum(weights[i] for i in range(count) if common & (1 << i)),
        "optimal_solutions_visited": optima_seen,
        "nodes_visited": visits, "ambiguous_objects": (union & ~common).bit_count(),
        "optimality_certified": True, "beam_truncated": False,
    }


def _evidence_key(row, adjacency, index, rows):
    neighbor_signatures = tuple(sorted((rows[j]["score"], rows[j]["macro_feature"],
                                        rows[j]["evidence_key"]) for j in adjacency[index]))
    return (-row["score"], row["macro_feature"], row["evidence_key"],
            -len(adjacency[index]), neighbor_signatures)


def _beam(rows, adjacency, width, tolerance):
    count = len(rows)
    keys = [_evidence_key(row, adjacency, i, rows) for i, row in enumerate(rows)]
    # An indistinguishable tied conflict could be silently broken by beam
    # truncation.  Remove all such peers before approximate optimization;
    # this is explicit abstention, not a fabricated unique optimum.
    abstain = {i for i in range(count) if any(keys[i] == keys[j] and rows[i]["score"] == rows[j]["score"]
                                                and j in adjacency[i] for j in range(count) if i != j)}
    ordered = sorted((i for i in range(count) if i not in abstain), key=lambda i: keys[i])
    states = [(0.0, frozenset())]
    truncated = False
    for index in ordered:
        options = states + [(score + rows[index]["score"], selected | {index})
                            for score, selected in states if not selected.intersection(adjacency[index])]
        options.sort(key=lambda state: -state[0])
        if len(options) > width:
            truncated = True
            # All states at a truncated objective tie must survive or none may
            # grant a supposedly decisive object.  Uncertified components are
            # marked below even when only clearly lower scores were dropped.
            cutoff = options[width - 1][0]
            states = [state for state in options if state[0] > cutoff + tolerance]
            ties = [state for state in options if abs(state[0] - cutoff) <= tolerance]
            if len(states) + len(ties) <= width:
                states.extend(ties)
            else:
                # The intersection has its own, generally lower objective.
                # Never assign the tied states' score to that intersection.
                shared = frozenset.intersection(*(s for _, s in ties))
                if not states:
                    states.append((sum(rows[i]["score"] for i in shared), shared))
        else:
            states = options
    best = max(score for score, _ in states)
    winners = [selected for score, selected in states if score >= best - tolerance]
    common = set.intersection(*(set(selection) for selection in winners))
    union = set.union(*(set(selection) for selection in winners))
    return common, {"method": "bounded_beam", "objective": best,
                    "selected_score": sum(rows[i]["score"] for i in common),
                    "optimal_solutions_visited": len(winners), "nodes_visited": None,
                    "ambiguous_objects": len(union - common), "symmetry_abstentions": len(abstain),
                    "optimality_certified": False, "beam_truncated": truncated}


def select_objects(candidates, *, exact_limit=32, beam_width=128, tie_tolerance=1e-9):
    """Select an independent set, returning only objects common to all best ties.

    This neither forces face coverage nor assumes topological connectedness.
    The approximate path is always labelled uncertified; never report its
    objective as an exact optimum.  All inputs are left untouched.
    """
    if exact_limit < 1 or beam_width < 1 or not 0 <= tie_tolerance < 1:
        raise ValueError("Invalid solver limits/tolerance")
    input_rows = list(candidates)
    unique = _deduplicate(input_rows)
    positive = [row for row in unique if row["score"] > tie_tolerance]
    selected = []
    component_rows = []
    for component, global_adjacency in _components(positive):
        indices = sorted(component)
        local = [positive[i] for i in indices]
        mapping = {original: local_index for local_index, original in enumerate(indices)}
        adjacency = [{mapping[j] for j in global_adjacency[i]} for i in indices]
        if len(local) <= exact_limit:
            chosen, detail = _exact(local, adjacency, tie_tolerance)
        else:
            chosen, detail = _beam(local, adjacency, beam_width, tie_tolerance)
        selected.extend(local[i] for i in chosen)
        component_rows.append({"candidate_count": len(local), "selected_count": len(chosen), **detail})
    selected.sort(key=lambda row: (-row["score"], row["macro_feature"], row["evidence_key"]))
    return {"selected_objects": [{"macro_feature": row["macro_feature"],
                                  "face_indices": list(row["face_indices"]), "score": row["score"]}
                                 for row in selected],
            "diagnostics": {"input_count": len(input_rows), "unique_count": len(unique),
                            "positive_count": len(positive), "negative_or_zero_rejected": len(unique) - len(positive),
                            "component_count": len(component_rows), "components": component_rows,
                            "all_optimality_certified": all(row["optimality_certified"] for row in component_rows),
                            "beam_truncation_count": sum(row["beam_truncated"] for row in component_rows)}}


__all__ = ["TRANSITIONS", "objects_conflict", "select_objects"]
