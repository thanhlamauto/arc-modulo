#!/usr/bin/env python3
"""Record and exercise every visible Kaggle CUDA device without assuming T4x2."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import time

import torch


OUTPUT = Path("/kaggle/working/dcs-results/canary/t4_canary.json")


def main() -> int:
    started = time.perf_counter()
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or count < 1:
        raise RuntimeError("no CUDA device is available")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        device = torch.device(f"cuda:{index}")
        left = torch.arange(256 * 256, dtype=torch.float32, device=device).reshape(256, 256)
        right = torch.eye(256, dtype=torch.float32, device=device)
        value = float((left @ right).sum().item())
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "tensor_check_sum": value,
            }
        )
    smi = subprocess.run(
        ["nvidia-smi", "-L"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    all_t4 = all("T4" in item["name"].upper() for item in devices)
    status = "PASS_T4X2" if count >= 2 and all_t4 else "PASS_ONE_T4" if count == 1 and all_t4 else "PASS_OTHER_CUDA"
    payload = {
        "schema_version": 1,
        "status": status,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": count,
        "devices": devices,
        "nvidia_smi_returncode": smi.returncode,
        "nvidia_smi_stdout": smi.stdout,
        "nvidia_smi_stderr": smi.stderr,
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DCS_T4_CANARY " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
