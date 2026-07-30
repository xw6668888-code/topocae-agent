from __future__ import annotations

"""CUDA-runtime-safe entry point for :mod:`train_p1_b_gnn_ablation`.

The installed PyTorch build can execute CUDA kernels but querying the optional
cuDNN DLL version raises ``cudnnGetVersion`` on this Windows image.  The P1-B
GraphSAGE models do not use convolution/cuDNN, so record that fact without
probing the unavailable symbol and delegate all training logic unchanged.
"""

import train_p1_b_gnn_ablation as implementation


def main() -> int:
    implementation.torch.backends.cudnn.version = lambda: "not_queried_no_convolution_dependency"
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
