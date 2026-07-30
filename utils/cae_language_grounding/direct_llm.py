"""Direct-LLM object/semantic-role selection baseline for P4-A1."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .llm_adapter import JSONInferenceOutcome, OpenAICompatibleJSONClient
from .memory import validate_feature_object_memory
from .records import assert_inference_input_safe


ROOT = Path(__file__).resolve().parents[2]
DIRECT_SCHEMA_PATH = ROOT / "schemas" / "p4_a1_direct_selection.schema.json"
DIRECT_PROMPT_PATH = ROOT / "prompts" / "p4_a1_direct_llm_v1.md"
WHOLE_PART_OPERATIONS = frozenset({"material_assignment", "global_mesh_size"})


@lru_cache(maxsize=1)
def _direct_validator() -> Draft202012Validator:
    schema = json.loads(DIRECT_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def validate_direct_selection(value: Any) -> dict[str, Any]:
    errors = []
    for error in sorted(
        _direct_validator().iter_errors(value),
        key=lambda item: list(item.absolute_path),
    ):
        errors.append(
            {
                "code": "E_DIRECT_SCHEMA",
                "path": "/" + "/".join(map(str, error.absolute_path)),
                "message": error.message,
            }
        )
    return {"valid": not errors, "errors": errors}


def sanitize_candidate_memory(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Expose candidates and descriptors, but no face indices or source identity."""

    validation = validate_feature_object_memory(memory)
    if not validation["valid"]:
        raise ValueError(f"Invalid feature-object memory: {validation['errors']}")
    candidates = []
    for obj in memory["objects"]:
        candidates.append(
            {
                "object_id": str(obj["object_id"]),
                "canonical_feature_type": str(
                    obj["canonical_feature_type"]
                ),
                "feature_family": str(obj["feature_family"]),
                "face_roles": [
                    {
                        "face_role": str(role["face_role"]),
                        "status": str(role["status"]),
                        "allowed_operations": list(
                            role["allowed_operations"]
                        ),
                        "region_kind": str(role["region_kind"]),
                        "requires_complete_instance": bool(
                            role["requires_complete_instance"]
                        ),
                    }
                    for role in obj["face_role_map"]
                ],
                "geometry_parameters": copy.deepcopy(
                    obj["geometry_parameters"]
                ),
                "descriptors": copy.deepcopy(obj["descriptors"]),
            }
        )
    result = {
        "schema_version": "topocae-direct-candidate-memory/1.0.0",
        "objects": candidates,
    }
    assert_inference_input_safe(result)
    return result


def make_direct_inference_input(
    *,
    text: str,
    language: str,
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "text": str(text),
        "language": str(language),
        "candidate_memory": sanitize_candidate_memory(memory),
    }
    assert_inference_input_safe(value)
    return value


def infer_direct_selection(
    client: OpenAICompatibleJSONClient,
    *,
    text: str,
    language: str,
    memory: Mapping[str, Any],
) -> JSONInferenceOutcome:
    schema = json.loads(DIRECT_SCHEMA_PATH.read_text(encoding="utf-8"))
    prompt = DIRECT_PROMPT_PATH.read_text(encoding="utf-8")
    inference_input = make_direct_inference_input(
        text=text,
        language=language,
        memory=memory,
    )
    return client.infer_json(
        inference_input=inference_input,
        system_prompt=prompt,
        schema_name="topocae_direct_selection_v1",
        schema=schema,
        validator=validate_direct_selection,
    )


def _rejection(
    code: str,
    message: str,
    *,
    illegal_id_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": "topocae-direct-grounding/1.0.0",
        "status": "INVALID",
        "error_code": code,
        "message": message,
        "operations": [],
        "verifier": {
            "status": "INVALID",
            "illegal_object_id_count": illegal_id_count,
            "illegal_face_id_count": 0,
            "silent_fallback_count": 0,
        },
    }


def expand_and_verify_direct_selection(
    selection: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand selected semantic roles locally; raw face IDs never come from LLM."""

    selection_validation = validate_direct_selection(selection)
    if not selection_validation["valid"]:
        return _rejection(
            "E_DIRECT_SCHEMA",
            json.dumps(selection_validation["errors"], ensure_ascii=False),
        )
    memory_validation = validate_feature_object_memory(memory)
    if not memory_validation["valid"]:
        raise ValueError(
            f"Invalid feature-object memory: {memory_validation['errors']}"
        )
    if selection["status"] != "VALID":
        return {
            "schema_version": "topocae-direct-grounding/1.0.0",
            "status": selection["status"],
            "error_code": selection["error_code"],
            "message": selection["message"],
            "operations": [],
            "verifier": {
                "status": selection["status"],
                "illegal_object_id_count": 0,
                "illegal_face_id_count": 0,
                "silent_fallback_count": 0,
            },
        }

    object_map = {str(item["object_id"]): item for item in memory["objects"]}
    operations = []
    selection_ids: set[str] = set()
    for item in selection["selections"]:
        selection_id = str(item["selection_id"])
        if selection_id in selection_ids:
            return _rejection(
                "E_DIRECT_SELECTION_ID_DUPLICATE",
                f"Duplicate selection_id: {selection_id}",
            )
        selection_ids.add(selection_id)
        operation_type = str(item["operation_type"])
        object_ids = [str(value) for value in item["selected_object_ids"]]
        face_role = item["face_role"]
        if operation_type in WHOLE_PART_OPERATIONS:
            if object_ids or face_role is not None:
                return _rejection(
                    "E_DIRECT_WHOLE_PART_CONTRACT",
                    "Whole-part operations cannot select objects or roles.",
                )
            operations.append(
                {
                    "selection_id": selection_id,
                    "operation_type": operation_type,
                    "target_contract": "whole_part",
                    "selected_object_ids": [],
                    "selected_face_indices": [],
                }
            )
            continue
        if not object_ids or not isinstance(face_role, str):
            return _rejection(
                "E_DIRECT_TARGET_REQUIRED",
                "Feature operations require objects and one semantic face role.",
            )
        unknown = sorted(set(object_ids) - set(object_map))
        if unknown:
            return _rejection(
                "E_DIRECT_INVALID_OBJECT_ID",
                "Unknown opaque object IDs were rejected.",
                illegal_id_count=len(unknown),
            )
        selected_faces: set[int] = set()
        for object_id in object_ids:
            obj = object_map[object_id]
            roles = {
                str(row["face_role"]): row for row in obj["face_role_map"]
            }
            role = roles.get(face_role)
            if role is None:
                return _rejection(
                    "E_DIRECT_INVALID_FACE_ROLE",
                    f"Role {face_role!r} is not declared for {object_id}.",
                )
            if (
                role["status"] != "AVAILABLE"
                or operation_type not in role["allowed_operations"]
            ):
                return _rejection(
                    "E_DIRECT_ROLE_UNAVAILABLE",
                    f"Role {face_role!r} is not selectable for {operation_type}.",
                )
            source_face_count = int(obj["topology"]["source_face_count"])
            faces = [int(value) for value in role["face_indices"]]
            if any(face < 0 or face >= source_face_count for face in faces):
                return _rejection(
                    "E_DIRECT_INTERNAL_FACE_RANGE",
                    "Candidate memory contains an out-of-range face index.",
                )
            selected_faces.update(faces)
        operations.append(
            {
                "selection_id": selection_id,
                "operation_type": operation_type,
                "target_contract": "feature_face_set",
                "selected_object_ids": sorted(set(object_ids)),
                "selected_face_indices": sorted(selected_faces),
                "face_role": face_role,
            }
        )

    return {
        "schema_version": "topocae-direct-grounding/1.0.0",
        "status": "VALID",
        "error_code": None,
        "message": "",
        "operations": operations,
        "verifier": {
            "status": "VALID",
            "illegal_object_id_count": 0,
            "illegal_face_id_count": 0,
            "silent_fallback_count": 0,
        },
    }


__all__ = [
    "expand_and_verify_direct_selection",
    "infer_direct_selection",
    "make_direct_inference_input",
    "sanitize_candidate_memory",
    "validate_direct_selection",
]
