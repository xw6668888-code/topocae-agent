from __future__ import annotations

"""Corrected artifact-manifest path entry for final P1-B delivery indexing."""

import build_p1_b_final_delivery_manifest as base


_wrong = base.ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "reports" / "artifact_manifest.json"
_correct = base.ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "manifests" / "artifact_manifest.json"
base.KEY_FILES = [_correct if path == _wrong else path for path in base.KEY_FILES]


if __name__ == "__main__":
    raise SystemExit(base.main())
