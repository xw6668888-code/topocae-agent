"""D0-train-only class/source precision evidence for a possible E3 prior.

This is an analysis artifact. No new scoring rule is trained or evaluated here.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from scipy.stats import beta

from e_joint_scorer import FOLDER, load_train
from g09_common import ARTIFACT, checked, descriptor, read, write_json


OUTPUT = ARTIFACT / "e_joint_source_priors/D0_train_seed_20260719.json"
GROUPS = ("R_final", "Q_only", "R_other")


def source_group(candidate):
    origins = {(row["branch"], row["stage"]) for row in candidate["provenance"]}
    if ("R", "final") in origins:
        return "R_final"
    if any(branch == "R" for branch, _ in origins):
        return "R_other"
    if origins and all(branch == "Q" for branch, _ in origins):
        return "Q_only"
    raise ValueError("Candidate has no registered R/Q provenance")


def posterior(pos, total):
    """Jeffreys Beta(1/2,1/2) posterior, useful for sparse classes."""
    if not 0 <= pos <= total:
        raise ValueError("Positive count outside candidate denominator")
    a, b = pos + .5, total - pos + .5
    return {"positive": pos, "candidates": total, "negative": total - pos,
            "raw_precision": pos / total if total else None,
            "jeffreys_mean": a / (a + b),
            "credible_95": [float(beta.ppf(.025, a, b)), float(beta.ppf(.975, a, b))]}


def analyze():
    if OUTPUT.exists():
        raise FileExistsError("Retain prior analysis instead of overwriting")
    seal, x, y, groups, counts = load_train()
    del x, y, groups
    outputs = {Path(item["path"]).name: checked(item) for item in seal["outputs"]}
    table = Counter()
    class_totals = Counter()
    group_totals = Counter()
    totals = Counter()
    with outputs["candidates.jsonl"].open(encoding="utf-8") as features, \
         outputs["labels.jsonl"].open(encoding="utf-8") as labels:
        for part, (fline, lline) in enumerate(zip(features, labels, strict=True)):
            frow, lrow = json.loads(fline), json.loads(lline)
            if frow["part_index"] != lrow["part_index"] or frow["part_index"] != part:
                raise ValueError("Train source-prior records are not paired")
            for candidate, label in zip(frow["candidates"], lrow["labels"], strict=True):
                macro, group, exact = candidate["macro_feature"], source_group(candidate), label["exact_positive"]
                for counter, key in ((table, (macro, group)), (class_totals, macro),
                                     (group_totals, group), (totals, "all")):
                    counter[(key, "candidate")] += 1
                    counter[(key, "positive")] += int(exact)
    if totals[("all", "candidate")] != counts["positive"] + counts["negative"] \
            or totals[("all", "positive")] != counts["positive"]:
        raise ValueError("Train source prior denominator changed")
    classes = sorted({key for (key, measure) in class_totals})
    result = {"schema": "g09_e_joint_train_source_priors_v1",
              "identity": seal["identity"], "train_seal": descriptor(FOLDER / "COMPLETE.json"),
              "group_rule": "R_final if R:final; else R_other if any R; else Q_only",
              "posterior": "Jeffreys Beta(0.5,0.5); 95% equal-tail credible interval",
              "overall": posterior(totals[("all", "positive")], totals[("all", "candidate")]),
              "by_source": {name: posterior(group_totals[(name, "positive")],
                                            group_totals[(name, "candidate")]) for name in GROUPS},
              "by_class": {name: {
                  "all": posterior(class_totals[(name, "positive")], class_totals[(name, "candidate")]),
                  **{group: posterior(table[((name, group), "positive")],
                                      table[((name, group), "candidate")]) for group in GROUPS},
              } for name in classes},
              "no_validation_or_engineering_labels": True, "inference_rule_changed": False}
    write_json(OUTPUT, result)
    return result


if __name__ == "__main__":
    data = analyze()
    print(json.dumps({"overall": data["overall"], "by_source": data["by_source"],
                      "classes": {cls: {origin: {key: row[origin][key] for key in (
                          "positive", "candidates", "jeffreys_mean", "credible_95")}
                          for origin in GROUPS} for cls, row in data["by_class"].items()}},
                     ensure_ascii=False), flush=True)
