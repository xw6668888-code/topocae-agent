"""Finalize the existing verified CAE-QueryGround formal outputs."""

from __future__ import annotations

import json

from utils.cae_query_ground.formal_finalize_release import finalize


if __name__ == "__main__":
    print(json.dumps(finalize(force=True), ensure_ascii=False, sort_keys=True))

