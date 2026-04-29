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
    parser = argparse.ArgumentParser(description="Run X-GATE claim-closure evaluation from Google Colab.")
    parser.add_argument("--drive-root", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--drift-samples", type=int, default=None)
    parser.add_argument("--adv-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
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

    results_root = Path(args.results_root) if args.results_root else layout.canonical_root
    output_root = Path(args.output_root) if args.output_root else layout.claim_closure_root
    if not results_root.exists():
        print(f"Canonical results root does not exist: {results_root}")
        print("Run colab_run_canonical.py first, or pass --results-root explicitly.")
        return 1

    eval_device = args.eval_device or ("cuda" if torch.cuda.is_available() else "cpu")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_claim_closure.py")),
        "--results-root",
        str(results_root),
        "--output-root",
        str(output_root),
        "--eval-device",
        eval_device,
    ]
    if args.n_runs is not None:
        cmd += ["--n-runs", str(args.n_runs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.drift_samples is not None:
        cmd += ["--drift-samples", str(args.drift_samples)]
    if args.adv_samples is not None:
        cmd += ["--adv-samples", str(args.adv_samples)]
    if args.max_eval_samples is not None:
        cmd += ["--max-eval-samples", str(args.max_eval_samples)]
    cmd += passthrough

    print(summarize_layout(layout))
    print(f"Canonical results  : {results_root}")
    print(f"Claim closure root : {output_root}")
    print(f"Command            : {shlex.join(cmd)}")

    subprocess.run(cmd, env=export_runtime_env(layout), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
