from __future__ import annotations

import train_p1_c_gnn_iteration1 as protocol
import train_p1_c_gnn_v2 as cuda_safe


protocol.protocol.baseline._environment = cuda_safe._environment_without_cudnn_probe


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
