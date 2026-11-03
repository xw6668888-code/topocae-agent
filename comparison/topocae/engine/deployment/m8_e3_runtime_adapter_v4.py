"""Process-local adapter for exact E3 model bytes in a relocated G10 bundle.

Only two protected comparisons are translated: checkpoint-embedded identities
are checked against their original digests, while current JSON descriptors are
checked against the relocated bundle. Model architecture/state checks remain.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from typing import Any, Iterator

from g10_common import digest, read_json, sha256
from m8_embedded_identity_v4 import (verify_materialized_call_sites,
                                     verify_q_binary_bound,
                                     verify_scorer_binary_bound)
from m8_portable_rebase_v4 import ClosureError


def _module(name: str, bundle: Path):
    module = importlib.import_module(name)
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(bundle.resolve()):
        raise ClosureError("runtime imported local module outside relocated bundle: " + name)
    return module


def _q_loader(module, plan: dict, original):
    def load(config, seed):
        if config == "full":
            return original(config, seed)
        freeze = module.read_json(module.REPORT / "CONFIGURATION_FREEZE.json")
        if config != "dense_instance" or config not in freeze["configs"] or seed not in freeze["seeds"]:
            raise ClosureError("relocated Q accepts only frozen dense_instance seed")
        run = module.ARTIFACT / "training_runs" / config / f"seed_{seed}"
        checkpoint = run / "best.pt"
        expected_metadata = [item for item in freeze["frozen_files"] if
                             item["path"] == (module.DATA / "metadata.json").resolve().as_posix()]
        if len(expected_metadata) != 1 or module.sha256(module.DATA / "metadata.json") != \
                expected_metadata[0]["sha256"]:
            raise ClosureError("relocated frozen normalization changed")
        if module.read_json(run / "STATUS.json").get("status") != "completed":
            raise ClosureError("relocated Q model training status is incomplete")
        checkpoint_identity = module.read_json(run / "metrics.json")["checkpoint"]
        if checkpoint.resolve().as_posix() != checkpoint_identity["path"] or \
                checkpoint.stat().st_size != checkpoint_identity["bytes"] or \
                module.sha256(checkpoint) != checkpoint_identity["sha256"]:
            raise ClosureError("relocated Q checkpoint descriptor differs")
        saved = module.torch.load(checkpoint, map_location="cpu", weights_only=False)
        verify_q_binary_bound(saved, plan)
        model = module.make_model()
        model.load_state_dict(saved["model_state_dict"])
        return model.eval(), module.read_json(run / "same_instance_threshold_selection.json"), checkpoint
    return load


def _scorer_loader(module, plan: dict):
    def load(folder=None):
        folder = Path(folder if folder is not None else module.OUTPUT)
        complete = module.read(folder / "COMPLETE.json")
        if complete.get("schema") != "g09_e_joint_scorer_v1" or complete.get("parts") != 7440:
            raise ClosureError("relocated E1 scorer has no complete D0 train seal")
        checkpoint_path = module.checked(complete["checkpoint"])
        expected_sha = plan["binary_bound_original_identity"]["E1_scorer_checkpoint_sha256"]
        if sha256(checkpoint_path) != expected_sha:
            raise ClosureError("relocated E1 scorer checkpoint bytes changed")
        saved = module.torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        verified = verify_scorer_binary_bound(saved, plan)
        if saved["feature_names"] != list(module.FEATURE_NAMES) or \
                verified["features"] != len(module.FEATURE_NAMES):
            raise ClosureError("relocated E1 source feature order differs")
        model = module.ObjectScorer(len(module.FEATURE_NAMES))
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        return model, module.torch.tensor(saved["feature_mean"]), module.torch.tensor(saved["feature_std"])
    return load


@contextmanager
def relocated_e3_runtime(bundle: Path, plan: dict, mapping: dict) -> Iterator[dict[str, Any]]:
    root = bundle.resolve()
    call_sites = verify_materialized_call_sites(root, plan, mapping)
    search = [root / "tools/aei_paper_goals_v1" / name for name in ("g09", "g07", "g03")]
    old_path = list(sys.path)
    patched = []
    try:
        for path in reversed(search):
            sys.path.insert(0, str(path))
        g03 = _module("inference", root)
        predictor = _module("topo_predictor", root)
        scorer = _module("e_joint_scorer", root)
        adapter = _module("e_v09_adapter", root)
        q_loader = _q_loader(g03, plan, g03.load_frozen)
        scorer_loader = _scorer_loader(scorer, plan)
        for module, name, value in ((predictor, "load_frozen", q_loader),
                                    (scorer, "load_scorer", scorer_loader),
                                    (adapter, "load_e1", scorer_loader)):
            patched.append((module, name, getattr(module, name)))
            setattr(module, name, value)
        for name in ("e_joint_calibration", "e_joint_eval"):
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "load_scorer"):
                if not Path(module.__file__).resolve().is_relative_to(root):
                    raise ClosureError("preloaded E3 module came from outside relocated bundle")
                patched.append((module, "load_scorer", module.load_scorer))
                module.load_scorer = scorer_loader
        yield {"schema": "g10_m8_e3_process_local_relocation_adapter_v4",
               "status": "INSTALLED_MODEL_LOADS_PENDING",
               "source_only_plan_digest": digest(plan),
               "mapping_digest": digest(mapping), "JSON_call_sites": call_sites,
               "translated_checks": ["Q_checkpoint_original_freeze_sha256",
                                      "E1_checkpoint_original_protocol_and_train_seal_digests"],
               "model_architecture_or_state_bypass": False,
               "actual_checkpoint_loads": 0, "actual_inferences": 0,
               "reproducibility_pass": False}
    finally:
        for module, name, value in reversed(patched):
            setattr(module, name, value)
        sys.path[:] = old_path
