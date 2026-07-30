from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .constants import ROOT
from .formal_planner import FINAL_DATASET_VERSION, FINAL_RECORD_VERSION
from .io import read_json, write_json


PILOT_RECORD_SCHEMA = ROOT / "schemas" / "cae_query_ground_record.schema.json"
PILOT_MANIFEST_SCHEMA = ROOT / "schemas" / "cae_query_ground_dataset_manifest.schema.json"
FINAL_RECORD_SCHEMA = ROOT / "schemas" / "cae_query_ground_record.v1.schema.json"
FINAL_MANIFEST_SCHEMA = ROOT / "schemas" / "cae_query_ground_dataset_manifest.v1.schema.json"


def build_final_schemas(*, force: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    record = copy.deepcopy(read_json(PILOT_RECORD_SCHEMA))
    record["$id"] = "https://topocae.local/schemas/cae_query_ground_record.v1.schema.json"
    record["title"] = "TopoCAE CAE-QueryGround Record v1 Frozen"
    record["properties"]["record_schema_version"]["const"] = FINAL_RECORD_VERSION

    manifest = copy.deepcopy(read_json(PILOT_MANIFEST_SCHEMA))
    manifest["$id"] = "https://topocae.local/schemas/cae_query_ground_dataset_manifest.v1.schema.json"
    manifest["title"] = "TopoCAE CAE-QueryGround Dataset Manifest v1 Frozen"
    manifest["properties"]["schema_version"]["const"] = (
        "cae-query-ground-dataset-manifest/1.0.0-h3.01"
    )
    manifest["properties"]["dataset_version"]["const"] = FINAL_DATASET_VERSION
    manifest["properties"]["status"]["const"] = "frozen"
    counts = manifest["properties"]["counts"]["properties"]
    counts["canonical_records"]["minimum"] = 20000
    counts["renderings"]["minimum"] = 60000
    review = manifest["properties"]["review_gate"]["properties"]
    review["status"]["const"] = "PASS"
    review["reviewed_rows"]["const"] = 120
    write_json(FINAL_RECORD_SCHEMA, record, force=force)
    write_json(FINAL_MANIFEST_SCHEMA, manifest, force=force)
    return record, manifest

