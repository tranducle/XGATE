from __future__ import annotations

import platform
import sys

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from colab_paths import build_layout, ensure_output_dirs, missing_dataset_files, summarize_layout


def main() -> int:
    layout = build_layout()
    ensure_output_dirs(layout)

    print("=" * 72)
    print("X-GATE Colab Setup Check")
    print("=" * 72)
    print(summarize_layout(layout))
    print("-" * 72)
    print(f"Python version     : {sys.version.split()[0]}")
    print(f"Platform           : {platform.platform()}")
    if torch is None:
        print("Torch              : NOT INSTALLED")
    else:
        print(f"Torch              : {torch.__version__}")
        print(f"CUDA available     : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU                : {torch.cuda.get_device_name(0)}")

    missing = missing_dataset_files(layout)
    print("-" * 72)
    if missing:
        print("Dataset status     : MISSING REQUIRED FILES")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("Dataset status     : READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
