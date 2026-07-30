"""H-C1 pilot bound to the coordinate-aligned clean rib/web generator."""

from __future__ import annotations

from p1_c import pilot_h_c1 as protocol
from p1_c import rib_web_v2 as generator


protocol.GENERATOR_REVISION = generator.GENERATOR_REVISION
protocol.SCENARIO_SPECS = generator.SCENARIO_SPECS
protocol.build_rib_web_sample = generator.build_rib_web_sample
protocol.scenario_keys = generator.scenario_keys

generate_h_c1 = protocol.generate_h_c1

__all__ = ["generate_h_c1"]
