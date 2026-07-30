from __future__ import annotations

import build_p1_c_candidate_training_package_v2 as protocol


implementation = protocol.implementation
_original_build_record = implementation._build_record


def _build_record_with_context_alias(graph, prediction, proposal, truths, source, **kwargs):
    if "derivation_kind" not in source:
        source = dict(source)
        source["derivation_kind"] = str(source.get("context", "unknown"))
    return _original_build_record(
        graph, prediction, proposal, truths, source, **kwargs
    )


implementation._build_record = _build_record_with_context_alias


def main() -> int:
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
