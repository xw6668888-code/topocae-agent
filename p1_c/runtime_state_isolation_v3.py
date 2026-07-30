"""Final import-order guard for captured function references in old revisions."""

from __future__ import annotations

from functools import wraps
from typing import Any

from p1_c.runtime_state_isolation_v2 import install as install_v2
from p1_c.scoped_bindings import scoped_module_bindings


_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    install_v2()

    from p1_c import formal_generator as generator_core
    from p1_c import formal_generator_v5
    from p1_c import formal_generator_v6
    from p1_c import rib_web_v5
    from p1_c import supported_wall_families as legacy_supported
    from p1_c import supported_wall_families_v2
    from p1_c import supported_wall_families_v3
    from p1_c import all_features_h_c2_v2

    # Some modules captured a function object with ``from ... import`` before
    # v1 wrapped the source module. Rebind those captures to guarded entrypoints.
    supported_wall_families_v2.build_rib_web_sample = (
        rib_web_v5.build_rib_web_sample
    )
    all_features_h_c2_v2.build_supported_wall_sample = (
        supported_wall_families_v3.build_supported_wall_sample
    )
    formal_generator_v6.build_supported_wall_sample = (
        supported_wall_families_v2.build_supported_wall_sample
    )

    legacy_bindings = {
        "build_supported_wall_sample": legacy_supported.build_supported_wall_sample,
        "_supported_host_parameters": generator_core._supported_host_parameters,
        "_supported_feature_centers": generator_core._supported_feature_centers,
    }
    current = formal_generator_v5.build_planned_result
    if not getattr(current, "_p1_c_legacy_scope_v3", False):
        @wraps(current)
        def guarded(plan: Any) -> Any:
            if getattr(plan, "host_family_kind", "") == "supported_wall_strict_concept":
                return current(plan)
            with scoped_module_bindings({generator_core: legacy_bindings}):
                return current(plan)

        guarded._p1_c_legacy_scope_v3 = True  # type: ignore[attr-defined]
        formal_generator_v5.build_planned_result = guarded

    _INSTALLED = True


install()


__all__ = ["install"]
