"""Leakage-safe language-example views over frozen train/val records."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .dataset import iter_canonical_records, iter_rendering_records


INFERENCE_FORBIDDEN_KEYS = frozenset(
    {
        "ground_truth",
        "target_instance_ids",
        "target_face_indices",
        "semantic_evidence",
        "uniqueness_contract",
        "intent_result",
        "canonical_record_id",
        "rendering_id",
        "sample_id",
        "graph_path",
        "geometry_path",
        "part_id",
        "face_id",
        "face_ids",
    }
)


@dataclass(frozen=True)
class LanguageExample:
    example_id: str
    canonical_record_id: str
    split: str
    text: str
    language: str
    difficulty_level: str
    oracle_intent_result: Mapping[str, Any]
    ground_truth: Mapping[str, Any]
    source_ref: Mapping[str, Any]

    def inference_input(self) -> dict[str, str]:
        value = {"text": self.text, "language": self.language}
        assert_inference_input_safe(value)
        return value


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def assert_inference_input_safe(value: Any) -> None:
    forbidden = sorted(INFERENCE_FORBIDDEN_KEYS & set(_walk_keys(value)))
    if forbidden:
        raise ValueError(
            "Inference input contains forbidden answer/identity fields: "
            + ", ".join(forbidden)
        )


def iter_language_examples(split: str) -> Iterable[LanguageExample]:
    canonical = {
        str(record["source_ref"]["canonical_record_id"]): record
        for record in iter_canonical_records(split)
    }
    for rendering in iter_rendering_records(split):
        if "query" in rendering:
            query = rendering["query"]
            canonical_id = str(rendering["source_ref"]["canonical_record_id"])
            source = rendering
        else:
            query = rendering
            canonical_id = str(rendering["canonical_record_id"])
            source = canonical[canonical_id]
        if canonical_id not in canonical:
            raise ValueError(
                f"Rendering refers to an unknown canonical record: {canonical_id}"
            )
        canonical_record = canonical[canonical_id]
        difficulty = str(
            query.get("difficulty_level")
            or canonical_record.get("query", {}).get("difficulty_level")
            or ""
        )
        yield LanguageExample(
            example_id=str(
                query.get("rendering_id")
                or rendering.get("rendering_id")
                or f"{canonical_id}:rendering"
            ),
            canonical_record_id=canonical_id,
            split=split,
            text=str(query.get("text", rendering.get("text", ""))),
            language=str(
                query.get("language", rendering.get("language", ""))
            ),
            difficulty_level=difficulty,
            oracle_intent_result=copy.deepcopy(rendering["intent_result"]),
            ground_truth=copy.deepcopy(
                rendering.get("ground_truth", canonical_record["ground_truth"])
            ),
            source_ref=copy.deepcopy(
                rendering.get("source_ref", canonical_record["source_ref"])
            ),
        )


__all__ = [
    "INFERENCE_FORBIDDEN_KEYS",
    "LanguageExample",
    "assert_inference_input_safe",
    "iter_language_examples",
]
