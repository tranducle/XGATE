from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import torch

from colab_paths import (
    build_layout,
    ensure_output_dirs,
    export_runtime_env,
    missing_dataset_files,
    summarize_layout,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run X-GATE ablation experiments from Google Colab.")
    parser.add_argument("--drive-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--drift-samples", type=int, default=None)
    parser.add_argument("--teacher-root", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    layout = build_layout(args.drive_root)
    ensure_output_dirs(layout)

    missing = missing_dataset_files(layout)
    if missing:
        print("Missing required dataset files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    teacher_root = Path(args.teacher_root) if args.teacher_root else layout.canonical_root
    if not teacher_root.exists():
        print(f"Teacher checkpoint root does not exist: {teacher_root}")
        print("Run colab_run_canonical.py first, or pass --teacher-root explicitly.")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_ablation_study.py")),
        "--results-root",
        str(layout.ablation_root),
        "--teacher-root",
        str(teacher_root),
        "--device",
        device,
        "--n-runs",
        str(args.n_runs),
    ]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.drift_samples is not None:
        cmd += ["--drift-samples", str(args.drift_samples)]
    if args.smoke_test:
        cmd.append("--smoke-test")
    cmd += passthrough

    print(summarize_layout(layout))
    print(f"Teacher root       : {teacher_root}")
    print(f"Command            : {shlex.join(cmd)}")

    subprocess.run(cmd, env=export_runtime_env(layout), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
