from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch

import train_p1_c_gnn as protocol


def _environment_without_cudnn_probe(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
        }
    nvidia_smi = None
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": "not_queried_no_convolution_dependency",
        "device": str(device),
        "gpu": gpu,
        "nvidia_smi": nvidia_smi,
        "numpy": np.__version__,
        "pid": os.getpid(),
        "cuda_smoke_matmul": True,
    }


protocol.baseline._environment = _environment_without_cudnn_probe


def main() -> int:
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
