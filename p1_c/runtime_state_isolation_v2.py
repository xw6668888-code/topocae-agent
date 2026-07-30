"""Compatibility refinements for the P1-C revision-state guard."""

from __future__ import annotations

from p1_c.runtime_state_isolation_v1 import install as install_v1


_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    install_v1()

    from p1_c import rib_web as rib_core
    from p1_c import rib_web_v2
    from p1_c import formal_generator_v7

    # The aligned L-corner kernel is the established clean legacy baseline.  It
    # is intentionally process-wide; only the later strict host dispatcher was
    # the contaminating revision.
    rib_core._l_corner_host = rib_web_v2._aligned_l_corner_host

    # v7 is deliberately fail-closed: each call repairs a tampered dependency
    # and leaves the authoritative dependency installed.  Do not restore a
    # malicious or stale pre-call binding after that repair.
    guarded = formal_generator_v7.build_planned_result
    original = getattr(guarded, "__wrapped__", None)
    if original is not None:
        formal_generator_v7.build_planned_result = original

    _INSTALLED = True


install()


__all__ = ["install"]
