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
    parser = argparse.ArgumentParser(description="Run canonical X-GATE experiments from Google Colab.")
    parser.add_argument("--drive-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=None)
    parser.add_argument("--student-epochs", type=int, default=None)
    parser.add_argument("--dl-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--drift-samples", type=int, default=None)
    parser.add_argument("--reuse-teacher-from", type=str, default=None)
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_all_experiments.py")),
        "--results-root",
        str(layout.canonical_root),
        "--device",
        device,
        "--n-runs",
        str(args.n_runs),
    ]
    if args.teacher_epochs is not None:
        cmd += ["--teacher-epochs", str(args.teacher_epochs)]
    if args.student_epochs is not None:
        cmd += ["--student-epochs", str(args.student_epochs)]
    if args.dl_epochs is not None:
        cmd += ["--dl-epochs", str(args.dl_epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.drift_samples is not None:
        cmd += ["--drift-samples", str(args.drift_samples)]
    if args.reuse_teacher_from is not None:
        cmd += ["--reuse-teacher-from", args.reuse_teacher_from]
    if args.smoke_test:
        cmd.append("--smoke-test")
    cmd += passthrough

    print(summarize_layout(layout))
    print(f"Command            : {shlex.join(cmd)}")

    subprocess.run(cmd, env=export_runtime_env(layout), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
