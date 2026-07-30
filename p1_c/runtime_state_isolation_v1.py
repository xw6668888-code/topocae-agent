"""Install import-order-independent wrappers for historical P1-C revisions.

Several early revision modules selected a geometry kernel by assigning to an
older module at import time.  That made results depend on which test happened
to be imported last.  This compatibility guard eagerly loads those revisions,
restores every shared module to its baseline state, and wraps revision entry
points with exception-safe scoped bindings.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from p1_c.scoped_bindings import scoped_module_bindings


_INSTALLED = False


def _replace(module: Any, name: str, factory: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
    original = getattr(module, name)
    if getattr(original, "_p1_c_state_isolated", False):
        return
    wrapped = factory(original)
    wrapped._p1_c_state_isolated = True  # type: ignore[attr-defined]
    setattr(module, name, wrapped)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from p1_c import rib_web as rib_core
    rib_baseline = {
        "_l_corner_host": rib_core._l_corner_host,
        "_two_support_host": rib_core._two_support_host,
        "_feature_profile": rib_core._feature_profile,
    }

    from p1_c import all_features as feature_core
    feature_baseline = {"_feature_tool": feature_core._feature_tool}

    from p1_c import formal_plan as plan_core
    plan_baseline = {
        "PLAN_REVISION": plan_core.PLAN_REVISION,
        "SEED_NAMESPACE": plan_core.SEED_NAMESPACE,
        "_secondary_stream": plan_core._secondary_stream,
        "build_plan": plan_core.build_plan,
        "audit_plan": plan_core.audit_plan,
        "compatibility_matrix": plan_core.compatibility_matrix,
    }

    from p1_c import formal_generator as generator_core
    generator_baseline = {
        "build_supported_wall_sample": generator_core.build_supported_wall_sample,
        "_supported_host_parameters": generator_core._supported_host_parameters,
        "_supported_feature_centers": generator_core._supported_feature_centers,
    }

    # Loading every participating revision now makes later imports side-effect
    # free because Python returns these cached, guarded modules.
    from p1_c import rib_web_v2, rib_web_v3, rib_web_v4, rib_web_v5
    from p1_c import supported_wall_families_v2, supported_wall_families_v3
    from p1_c import all_features_sweep_v2, all_features_h_c2_v3
    from p1_c import formal_plan_v4, formal_plan_v5

    plan_v5_revision = formal_plan_v5.PLAN_REVISION
    plan_v5_seed = formal_plan_v5.SEED_NAMESPACE
    from p1_c import formal_plan_v6
    from p1_c import formal_generator_v3, formal_generator_v4
    from p1_c import formal_generator_v5, formal_generator_v6, formal_generator_v7

    # Undo all import-time assignments before installing scoped call wrappers.
    for name, value in rib_baseline.items():
        setattr(rib_core, name, value)
    for name, value in feature_baseline.items():
        setattr(feature_core, name, value)
    for name, value in plan_baseline.items():
        setattr(plan_core, name, value)
    formal_plan_v5.PLAN_REVISION = plan_v5_revision
    formal_plan_v5.SEED_NAMESPACE = plan_v5_seed
    for name, value in generator_baseline.items():
        setattr(generator_core, name, value)

    def rib_v2_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {rib_core: {"_l_corner_host": rib_web_v2._aligned_l_corner_host}}
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(rib_web_v2, "build_rib_web_sample", rib_v2_factory)

    def rib_v3_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(scenario_key: str, seed: int) -> Any:
            existed = scenario_key in rib_core.SCENARIO_BY_KEY
            previous = rib_core.SCENARIO_BY_KEY.get(scenario_key)
            try:
                with scoped_module_bindings(
                    {
                        rib_core: {
                            "_l_corner_host": rib_web_v2._aligned_l_corner_host,
                            "_two_support_host": rib_web_v3._strict_host_dispatch,
                            "_feature_profile": rib_web_v3._strict_feature_profile,
                        }
                    }
                ):
                    return original(scenario_key, seed)
            finally:
                if existed:
                    rib_core.SCENARIO_BY_KEY[scenario_key] = previous
                else:
                    rib_core.SCENARIO_BY_KEY.pop(scenario_key, None)
        return guarded

    _replace(rib_web_v3, "build_rib_web_sample", rib_v3_factory)

    def rib_v4_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {
                    rib_web_v3: {
                        "_web_tapered_host": rib_web_v4._web_tapered_host,
                        "_web_buttressed_host": rib_web_v4._web_buttressed_host,
                    }
                }
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(rib_web_v4, "build_rib_web_sample", rib_v4_factory)

    def rib_v5_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {rib_web_v4: {"_web_buttressed_host": rib_web_v5._web_buttressed_host}}
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(rib_web_v5, "build_rib_web_sample", rib_v5_factory)

    def supported_v3_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {
                    supported_wall_families_v2: {
                        "build_rib_web_sample": rib_web_v5.build_rib_web_sample,
                        "RIB_WEB_GENERATOR_REVISION": rib_web_v5.GENERATOR_REVISION,
                    }
                }
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(supported_wall_families_v3, "build_supported_wall_sample", supported_v3_factory)

    def sweep_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {feature_core: {"_feature_tool": all_features_sweep_v2._feature_tool}}
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(all_features_sweep_v2, "build_feature_sample", sweep_factory)

    def h_c2_v3_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings({feature_core: feature_baseline}):
                return original(*args, **kwargs)
        return guarded

    _replace(all_features_h_c2_v3, "build_feature_sample", h_c2_v3_factory)

    def plan_v4_build_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {
                    plan_core: {
                        "PLAN_REVISION": formal_plan_v4.PLAN_REVISION,
                        "_secondary_stream": formal_plan_v4._secondary_stream,
                        "compatibility_matrix": formal_plan_v4.compatibility_matrix,
                        "audit_plan": plan_baseline["audit_plan"],
                    }
                }
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(formal_plan_v4, "build_plan", plan_v4_build_factory)

    def plan_v4_audit_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {plan_core: {"compatibility_matrix": formal_plan_v4.compatibility_matrix}}
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(formal_plan_v4, "audit_plan", plan_v4_audit_factory)

    def plan_v6_build_factory(_original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(_original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {
                    formal_plan_v5: {
                        "PLAN_REVISION": formal_plan_v6.PLAN_REVISION,
                        "SEED_NAMESPACE": formal_plan_v6.SEED_NAMESPACE,
                    }
                }
            ):
                return formal_plan_v5.build_plan(*args, **kwargs)
        return guarded

    _replace(formal_plan_v6, "build_plan", plan_v6_build_factory)

    def plan_v6_audit_factory(_original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(_original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {
                    formal_plan_v5: {
                        "PLAN_REVISION": formal_plan_v6.PLAN_REVISION,
                        "SEED_NAMESPACE": formal_plan_v6.SEED_NAMESPACE,
                    }
                }
            ):
                return formal_plan_v5.audit_plan(*args, **kwargs)
        return guarded

    _replace(formal_plan_v6, "audit_plan", plan_v6_audit_factory)

    def generator_v3_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {feature_core: {"_select_outer_top_edge": formal_generator_v3._select_clean_outer_top_edge}}
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(formal_generator_v3, "build_planned_result", generator_v3_factory)

    def generator_v4_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {feature_core: {"_apply_transition": formal_generator_v4._apply_clean_transition}}
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(formal_generator_v4, "build_planned_result", generator_v4_factory)

    def generator_v5_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {formal_generator_v4: {"_transition_candidates": formal_generator_v5._all_geometric_outer_linear_edges}}
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(formal_generator_v5, "build_planned_result", generator_v5_factory)

    def generator_v6_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {
                    generator_core: {
                        "build_supported_wall_sample": formal_generator_v6.build_supported_wall_sample,
                        "_supported_host_parameters": formal_generator_v6._strict_supported_host_parameters,
                        "_supported_feature_centers": formal_generator_v6._strict_supported_feature_centers,
                    }
                }
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(formal_generator_v6, "build_planned_result", generator_v6_factory)

    def generator_v7_factory(original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            with scoped_module_bindings(
                {
                    feature_core: feature_baseline,
                    generator_core: generator_baseline,
                    formal_generator_v4: {
                        "_transition_candidates": formal_generator_v4._transition_candidates,
                    },
                }
            ):
                return original(*args, **kwargs)
        return guarded

    _replace(formal_generator_v7, "build_planned_result", generator_v7_factory)

    _INSTALLED = True


install()


__all__ = ["install"]
