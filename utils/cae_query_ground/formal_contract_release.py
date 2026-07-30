from __future__ import annotations

from .formal_contract import (
    FINAL_MANIFEST_SCHEMA,
    build_final_schemas as build_base,
)
from .io import read_json, write_json


def build_final_schemas(*, force: bool):
    record, manifest = build_base(force=force)
    manifest = read_json(FINAL_MANIFEST_SCHEMA)
    scope = manifest["properties"]["scope_guards"]
    scope["properties"]["formal_scale_generated"]["const"] = True
    write_json(FINAL_MANIFEST_SCHEMA, manifest, force=True)
    return record, manifest

