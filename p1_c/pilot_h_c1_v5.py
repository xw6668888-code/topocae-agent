"""Regenerated H-C1 after full-height web interface and cleanup repair."""

from __future__ import annotations

from p1_c import pilot_h_c1_v3 as protocol
from p1_c import rib_web_v5 as generator


protocol.generator = generator
protocol.protocol.GENERATOR_REVISION = generator.GENERATOR_REVISION
protocol.protocol.SCENARIO_SPECS = generator.SCENARIO_SPECS
protocol.protocol.build_rib_web_sample = generator.build_rib_web_sample
protocol.protocol.scenario_keys = generator.scenario_keys
protocol.PILOT_SEED_BASE = 2026087000
protocol.CANONICAL_SEED_BASE = 2026088000

generate_h_c1 = protocol.generate_h_c1


__all__ = ["generate_h_c1"]
