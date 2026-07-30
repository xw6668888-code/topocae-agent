from __future__ import annotations

import generate_p1_c_clean_context_7200 as protocol
from p1_c import formal_generator_v5 as generator


protocol.GENERATOR_REVISION = generator.GENERATOR_REVISION
protocol.build_planned_result = generator.build_planned_result
protocol.transformed_shape_and_audit = generator.transformed_shape_and_audit


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
