"""Frozen P1-C formal plan after the strict H-C1/H-C2 approvals."""

from __future__ import annotations

from p1_c import formal_plan as core
from p1_c import formal_plan_v5 as revision5


PLAN_REVISION = "p1_c_clean_context_strict_fail_closed_plan_r6"
SEED_NAMESPACE = 202609300000

# The r5 plan already contains the approved class/context/host allocation.  A
# new namespace is mandatory because no geometry from the invalidated formal
# run may be reused.  The base writer resolves these globals at call time.
revision5.PLAN_REVISION = PLAN_REVISION
revision5.SEED_NAMESPACE = SEED_NAMESPACE
core.PLAN_REVISION = PLAN_REVISION
core.SEED_NAMESPACE = SEED_NAMESPACE
core.build_plan = revision5.build_plan
core.audit_plan = revision5.audit_plan
core.compatibility_matrix = revision5.compatibility_matrix

CONTEXT_QUOTAS = revision5.CONTEXT_QUOTAS
FORBIDDEN_CLASS_PAIRS = revision5.FORBIDDEN_CLASS_PAIRS
PlannedSample = revision5.PlannedSample
audit_plan = revision5.audit_plan
build_plan = revision5.build_plan
compatibility_matrix = revision5.compatibility_matrix
load_plan = core.load_plan
sha256_file = core.sha256_file
write_plan = core.write_plan


__all__ = [
    "CONTEXT_QUOTAS",
    "FORBIDDEN_CLASS_PAIRS",
    "PLAN_REVISION",
    "SEED_NAMESPACE",
    "PlannedSample",
    "audit_plan",
    "build_plan",
    "compatibility_matrix",
    "load_plan",
    "sha256_file",
    "write_plan",
]
