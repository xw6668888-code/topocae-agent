"""Independent, relocated recount of scored T10 channels and registered M5 CIs.

Only a post-score, hashed M8 input index is accepted. This module never runs a
model, geometry kernel, solver or language API, and never opens project truth
before the one-time scoring gate has already been recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


TIERS = ("T10-I", "T10-P", "T10-S")
SEEDS = (20260718, 20260719, 20260720)
CLASSES = (
    "through_hole_round", "blind_hole_round", "stepped_hole_round",
    "counterbore_hole", "countersink_hole", "obround_hole", "slot_u",
    "slot_rect", "pocket_rect", "pocket_circular", "cylinder_boss", "rib",
    "web", "rectangular_boss", "fillet", "chamfer", "slot_v", "slot_o",
)
FACE_CLASSES = ("stock",) + CLASSES
METHODS = ("E3", "R", "effective_AAGNet", "MFTReNet-SI-D0")
COMPARISONS = (
    ("E3_minus_R", "R"),
    ("E3_minus_effective_AAGNet", "effective_AAGNet"),
    ("E3_minus_recent_instance_baseline", "MFTReNet-SI-D0"),
)


class RecountError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(root: Path, row: dict) -> Path:
    name = row.get("relative_path") if isinstance(row, dict) else None
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/") or ":" in name or \
            any(part in (".", "..") for part in name.split("/")):
        raise RecountError("recount asset needs a safe POSIX bundle-relative path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or \
            path.stat().st_size != row.get("bytes") or _sha(path) != row.get("sha256"):
        raise RecountError("missing, escaped or changed recount asset: " + name)
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _counter(objects: list[dict], face_count: int | None = None) -> Counter:
    values = Counter()
    for item in objects:
        name, faces = item["class"], item["faces"]
        if name not in CLASSES or not isinstance(faces, list) or not faces or \
                any(type(face) is not int or face < 0 or
                    (face_count is not None and face >= face_count) for face in faces) or \
                len(faces) != len(set(faces)):
            raise RecountError("invalid exact-instance object")
        values[name, tuple(sorted(faces))] += 1
    return values


def _counts(predicted: list[dict], truth: list[dict]) -> np.ndarray:
    left, right = _counter(predicted), _counter(truth)
    result = np.zeros((len(CLASSES), 3), dtype=np.int64)
    for index, name in enumerate(CLASSES):
        pred = Counter({key: amount for key, amount in left.items() if key[0] == name})
        actual = Counter({key: amount for key, amount in right.items() if key[0] == name})
        tp = sum((pred & actual).values())
        result[index] = tp, sum(pred.values()) - tp, sum(actual.values()) - tp
    return result


def _ratio(n: int, d: int) -> float:
    return n / d if d else 0.0


def _summary(counts: np.ndarray, classes: tuple[str, ...] = CLASSES) -> dict:
    per_class = {}
    for name, (tp, fp, fn) in zip(classes, counts.sum(axis=0)):
        tp, fp, fn = int(tp), int(fp), int(fn)
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn,
                           "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                           "f1": _ratio(2 * tp, 2 * tp + fp + fn)}
    tp, fp, fn = (int(value) for value in counts.sum(axis=(0, 1)))
    return {"tp": tp, "fp": fp, "fn": fn,
            "micro_precision": _ratio(tp, tp + fp), "micro_recall": _ratio(tp, tp + fn),
            "micro_f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "macro_f1": statistics.mean(item["f1"] for item in per_class.values()),
            "per_class": per_class}


def _face_summary(labels: list[list[str | None]], actual: list[list[str]]) -> dict:
    counts = np.zeros((len(FACE_CLASSES), 3), dtype=np.int64)
    correct = total = 0
    for pred, truth in zip(labels, actual):
        if not pred:
            pred = [None] * len(truth)
        if len(pred) != len(truth) or any(value not in FACE_CLASSES for value in truth) or \
                any(value is not None and value not in FACE_CLASSES for value in pred):
            raise RecountError("face labels violate the frozen ontology or denominator")
        total += len(truth)
        correct += sum(p == t for p, t in zip(pred, truth))
        for index, name in enumerate(FACE_CLASSES):
            counts[index, 0] += sum(p == name and t == name for p, t in zip(pred, truth))
            counts[index, 1] += sum(p == name and t != name for p, t in zip(pred, truth))
            counts[index, 2] += sum(p != name and t == name for p, t in zip(pred, truth))
    result = _summary(counts[None, ...], FACE_CLASSES)
    return {**result, "correct": correct, "total": total, "accuracy": _ratio(correct, total)}


def _subset(predictions: list[dict], truths: list[dict], selected: list[int]) -> dict:
    for i in selected:
        _counter(predictions[i]["objects"], truths[i]["face_count"])
        _counter(truths[i]["objects"], truths[i]["face_count"])
    counts = np.stack([_counts(predictions[i]["objects"], truths[i]["objects"]) for i in selected])
    exact = sum(_counter(predictions[i]["objects"]) == _counter(truths[i]["objects"])
                for i in selected)
    return {"parts": len(selected),
            "inference_failures": sum(predictions[i]["status"] == "failed" for i in selected),
            "whole_part_exact_count": exact, "whole_part_exact_rate": _ratio(exact, len(selected)),
            "exact_instances": _summary(counts),
            "faces": _face_summary([predictions[i]["face_macro_classes"] for i in selected],
                                   [truths[i]["face_macro_classes"] for i in selected])}


def _equal(expected, actual, *, name: str) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict) and set(expected) == set(actual):
        for key in expected:
            _equal(expected[key], actual[key], name=f"{name}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        for index, (left, right) in enumerate(zip(expected, actual)):
            _equal(left, right, name=f"{name}[{index}]")
    elif isinstance(expected, float) and isinstance(actual, (int, float)) and \
            math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12):
        return
    elif expected != actual or type(expected) is not type(actual):
        raise RecountError("independent recount differs: " + name)


def _macro(array: np.ndarray) -> float:
    total = array.sum(axis=0)
    denominator = 2 * total[:, 0] + total[:, 1] + total[:, 2]
    f1 = np.divide(2 * total[:, 0], denominator, out=np.zeros(len(CLASSES)), where=denominator != 0)
    return float(f1.mean())


def _bootstrap(counts: np.ndarray, rows: list[dict], effective: str,
               *, replicates: int = 5000, per_tier: int = 400) -> dict:
    if counts.shape != (len(METHODS), len(SEEDS), len(rows), len(CLASSES), 3):
        raise RecountError("CI source does not cover four methods, three seeds and all fixed parts")
    strata = {tier: np.asarray([i for i, row in enumerate(rows) if row["tier"] == tier], dtype=np.int64)
              for tier in TIERS}
    if any(len(indices) != per_tier for indices in strata.values()):
        raise RecountError("CI stratification differs from the preregistered T10 denominator")
    groups = {}
    for tier in TIERS:
        by_key = {}
        for i in strata[tier]:
            row = rows[int(i)]
            key = row.get("mother_id") or row.get("structure_signature")
            if key in (None, ""):
                raise RecountError("dependency CI needs mother or structure identity")
            by_key.setdefault(str(key), []).append(int(i))
        groups[tier] = [np.asarray(by_key[key], dtype=np.int64) for key in sorted(by_key)]

    def values(indices: np.ndarray) -> dict[str, float]:
        return {name: statistics.mean(_macro(counts[j, seed, indices]) for seed in range(len(SEEDS)))
                for j, name in enumerate(METHODS)}

    point = values(np.arange(len(rows), dtype=np.int64))
    primary = {name: [] for name, _ in COMPARISONS}
    dependency = {name: [] for name, _ in COMPARISONS}
    group_sizes = {tier: [] for tier in TIERS}
    part_rng = np.random.default_rng(2026091501)
    group_rng = np.random.default_rng(2026091502)
    for _ in range(replicates):
        chosen = np.concatenate([part_rng.choice(strata[tier], size=per_tier, replace=True)
                                 for tier in TIERS])
        sampled = values(chosen)
        dependent = []
        for tier in TIERS:
            picked = []
            while len(picked) < per_tier:
                picked.extend(groups[tier][int(group_rng.integers(len(groups[tier])))].tolist())
            group_sizes[tier].append(len(picked))
            dependent.extend(picked)
        group_sampled = values(np.asarray(dependent, dtype=np.int64))
        for name, other in COMPARISONS:
            primary[name].append(sampled["E3"] - sampled[other])
            dependency[name].append(group_sampled["E3"] - group_sampled[other])

    def interval(values: list[float]) -> dict:
        limits = np.quantile(values, [0.025, 0.975, 1 / 120, 119 / 120], method="linear")
        return {"percentile_95": [float(limits[0]), float(limits[1])],
                "bonferroni_simultaneous_98_333333": [float(limits[2]), float(limits[3])],
                "replicates": [float(value) for value in values]}

    return {"method_seed_mean_macro_f1": point,
            "comparisons": {name: {"observed_difference": point["E3"] - point[other],
                                  "part_bootstrap": interval(primary[name]),
                                  "dependency_group_sensitivity_only": interval(dependency[name])}
                            for name, other in COMPARISONS},
            "dependency_replicate_tier_sizes": group_sizes,
            "seed_per_method": {name: [_macro(counts[j, seed]) for seed in range(len(SEEDS))]
                                for j, name in enumerate(METHODS)},
            "class_support": {name: int(counts[0, 0, :, index, 0].sum() +
                                        counts[0, 0, :, index, 2].sum())
                              for index, name in enumerate(CLASSES)}}


def recount(bundle_root: Path, *, expected_denominator: int = 1200,
            replicates: int = 5000, per_tier: int = 400) -> dict:
    root = bundle_root.resolve(strict=True)
    index = _json(root / "M8_OFFLINE_RECOUNT_INPUTS.json")
    if index.get("schema") != "g10_m8_portable_recount_inputs_v3" or \
            index.get("status") != "exported_only_after_one_time_scoring" or \
            index.get("sealed_gate_verified") is not True or \
            index.get("fixed_denominator") != expected_denominator or \
            index.get("part_replicates") != replicates or \
            index.get("dependency_replicates") != replicates:
        raise RecountError("post-score sealed T10 recount input index is absent/incomplete")
    if expected_denominator != 3 * per_tier or per_tier < 1 or replicates < 1:
        raise RecountError("invalid fixed recount denominator/registered CI schedule")
    assets = index.get("assets")
    if not isinstance(assets, dict):
        raise RecountError("recount input asset ledger missing")
    source = _json(_asset(root, assets["source_only_manifest"]))
    rows = source.get("rows")
    if source.get("schema") != "g10_t10_source_only_manifest_v1" or not isinstance(rows, list) or \
            len(rows) != expected_denominator or len({row.get("sample_id") for row in rows}) != len(rows):
        raise RecountError("source-only sample denominator/order differs")
    targets = assets.get("targets")
    if not isinstance(targets, list) or len(targets) != len(rows):
        raise RecountError("post-score targets do not cover every sealed part")
    truths = []
    for row, item in zip(rows, targets):
        if item.get("sample_id") != row["sample_id"]:
            raise RecountError("target and source sample order differ")
        target = _json(_asset(root, item["asset"]))
        if target.get("sample_id") != row["sample_id"] or \
                target.get("source_step_sha256") != row["step"]["sha256"] or \
                target.get("face_count") != row["face_count"] or \
                len(target.get("face_macro_classes", [])) != row["face_count"]:
            raise RecountError("target changed fixed source identity or face denominator")
        _counter(target["objects"], row["face_count"])
        truths.append(target)

    predictions = assets.get("predictions")
    recorded = _json(_asset(root, assets["recorded_main_results"]))
    registered = _json(_asset(root, assets["recorded_bootstrap_results"]))
    effective = index.get("effective_aagnet_output")
    if effective not in ("AAGNet18-SI", "AAGNet18-SI-wide80", "AAGNet18-SI+own-selector") or \
            registered.get("effective_aagnet_output") != effective or \
            recorded.get("fixed_denominator") != expected_denominator or \
            registered.get("fixed_denominator") != expected_denominator or \
            not isinstance(predictions, dict) or set(predictions) != set(recorded.get("channels", {})):
        raise RecountError("main channels, preselected AAGNet or scored denominator differ")
    all_results = {}
    counts = np.zeros((len(METHODS), len(SEEDS), len(rows), len(CLASSES), 3), dtype=np.int64)
    selected = ("E3", "R", effective, "MFTReNet-SI-D0")
    for key in sorted(predictions):
        output, seed_text = key.rsplit("|", 1)
        seed = int(seed_text)
        if seed not in SEEDS:
            raise RecountError("unknown scored prediction seed")
        lines = _asset(root, predictions[key]).read_text(encoding="utf-8").splitlines()
        channel = [json.loads(line) for line in lines]
        if len(channel) != len(rows) or [item.get("sample_id") for item in channel] != \
                [row["sample_id"] for row in rows] or \
                any(item.get("output") != output or item.get("seed") != seed for item in channel):
            raise RecountError("sealed prediction channel identity/order differs")
        values = [item["prediction"] for item in channel]
        for source_row, value in zip(rows, values):
            if value.get("status") not in ("ok", "failed") or \
                    len(value.get("face_macro_classes", [])) not in (0, source_row["face_count"]):
                raise RecountError("prediction failure/face denominator invalid")
        overall = _subset(values, truths, list(range(len(rows))))
        per_tier_results = {tier: _subset(values, truths, [i for i, row in enumerate(rows)
                                                           if row["tier"] == tier]) for tier in TIERS}
        result = {"overall": overall, "per_tier": per_tier_results}
        _equal(recorded["channels"][key], result, name="main." + key)
        all_results[key] = result
        if output in selected:
            method = selected.index(output)
            seed_index = SEEDS.index(seed)
            for i, (value, target) in enumerate(zip(values, truths)):
                counts[method, seed_index, i] = _counts(value["objects"], target["objects"])
    if any(f"{output}|{seed}" not in all_results for output in selected for seed in SEEDS):
        raise RecountError("registered paired CI has missing method/seed channels")
    endpoint = {"macro_f1": lambda result: result["overall"]["exact_instances"]["macro_f1"],
                "micro_f1": lambda result: result["overall"]["exact_instances"]["micro_f1"],
                "whole_part_exact_rate": lambda result: result["overall"]["whole_part_exact_rate"],
                "face_accuracy": lambda result: result["overall"]["faces"]["accuracy"],
                "face_macro_f1": lambda result: result["overall"]["faces"]["macro_f1"]}
    summaries = {output: {name: {"mean": statistics.mean(function(all_results[f"{output}|{seed}"])
                                                       for seed in SEEDS),
                                "sample_standard_deviation": statistics.stdev(
                                    function(all_results[f"{output}|{seed}"]) for seed in SEEDS)}
                          for name, function in endpoint.items()}
                 for output in sorted({key.rsplit("|", 1)[0] for key in all_results})}
    _equal(recorded["three_seed_summary"], summaries, name="main.three_seed_summary")
    ci = _bootstrap(counts, rows, effective, replicates=replicates, per_tier=per_tier)
    _equal(registered["result"], ci, name="paired_CI")
    return {"schema": "g10_m8_independent_recount_acceptance_v3",
            "status": "PASS" if expected_denominator == 1200 and replicates == 5000 and per_tier == 400
                      else "PASS_SYNTHETIC_FIXTURE_ONLY_NOT_FORMAL_REPLAY",
            "full_main_table_and_CI_match": True,
            "paper_MAIN_RESULTS_csv_verified": False,
            "scope_of_main_table_match": "all_one_time_T10_scored_method_seed_tier_channels_and_seed_summaries",
            "sealed_T10_gate_verified": True,
            "fixed_denominator": expected_denominator,
            "channels_recomputed": len(all_results), "paired_CI_replicates": replicates,
            "actual_offline_recounts": 1, "model_or_solver_executed": False,
            "reproducibility_pass": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = recount(args.bundle_root)
    output = args.output_dir.resolve()
    if output.exists():
        raise RecountError("preserve one immutable recount output; use a new directory for corrections")
    output.mkdir(parents=True, exist_ok=False)
    (output / "M8_RECOUNT_ACCEPTANCE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
