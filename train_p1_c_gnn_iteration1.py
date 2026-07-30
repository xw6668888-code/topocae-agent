from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

import train_p1_c_gnn as protocol


ROOT = Path(__file__).resolve().parent
VARIANT = "iteration1_capped_instance_loss"
_ORIGINAL_COLLATE = protocol._collate


def _capped_instance_collate(records):
    batch = _ORIGINAL_COLLATE(records)
    weights = []
    for record in records:
        instance_ids = [int(value) for value in record["instance_ids"].tolist()]
        groups = defaultdict(list)
        for index, instance_id in enumerate(instance_ids):
            groups[instance_id].append(index)
        sample_weights = [0.0] * len(instance_ids)
        stock = groups.get(0, [])
        feature_groups = [groups[key] for key in sorted(groups) if key > 0]
        if stock and feature_groups:
            for index in stock:
                sample_weights[index] = 0.5 / len(stock)
            per_instance_total = 0.5 / len(feature_groups)
            for indices in feature_groups:
                for index in indices:
                    sample_weights[index] = per_instance_total / len(indices)
        else:
            active_groups = [indices for indices in groups.values() if indices]
            for indices in active_groups:
                for index in indices:
                    sample_weights[index] = 1.0 / len(active_groups) / len(indices)
        if abs(sum(sample_weights) - 1.0) > 1e-6:
            raise AssertionError("iteration-1 sample node weights do not sum to one")
        weights.append(torch.tensor(sample_weights, dtype=torch.float32))
    batch["node_weights"] = torch.cat(weights)
    return batch


protocol.VARIANTS = (*protocol.VARIANTS, VARIANT)
protocol.PACKAGE_BY_VARIANT[VARIANT] = "clean_7200"
protocol.PLAN = ROOT / "reports" / "p1_c_iterations_v1" / "iteration_1_hypothesis.json"
protocol._collate = _capped_instance_collate


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
