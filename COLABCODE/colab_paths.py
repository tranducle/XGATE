from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/XGATE_COLAB")


@dataclass(frozen=True)
class ColabLayout:
    code_dir: Path
    drive_root: Path
    dataset_dir: Path
    runs_root: Path
    canonical_root: Path
    ablation_root: Path
    claim_closure_root: Path


def _path_from_env(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else fallback


def resolve_drive_root(drive_root: str | None = None) -> Path:
    if drive_root:
        return Path(drive_root).expanduser()
    return _path_from_env("XGATE_COLAB_ROOT", DEFAULT_DRIVE_ROOT)


def build_layout(drive_root: str | None = None) -> ColabLayout:
    code_dir = Path(__file__).resolve().parent
    resolved_drive_root = resolve_drive_root(drive_root)
    dataset_dir = _path_from_env(
        "XGATE_DATASET_DIR",
        resolved_drive_root / "DATASET" / "processed",
    )
    runs_root = _path_from_env("XGATE_RUNS_ROOT", resolved_drive_root / "RUNS")
    return ColabLayout(
        code_dir=code_dir,
        drive_root=resolved_drive_root,
        dataset_dir=dataset_dir,
        runs_root=runs_root,
        canonical_root=runs_root / "canonical_multirun",
        ablation_root=runs_root / "ablation",
        claim_closure_root=runs_root / "claim_closure",
    )


def ensure_output_dirs(layout: ColabLayout) -> None:
    for path in (
        layout.drive_root,
        layout.runs_root,
        layout.canonical_root,
        layout.ablation_root,
        layout.claim_closure_root,
    ):
        path.mkdir(parents=True, exist_ok=True)


def export_runtime_env(layout: ColabLayout) -> dict[str, str]:
    env = os.environ.copy()
    env["XGATE_COLAB_ROOT"] = str(layout.drive_root)
    env["XGATE_DATASET_DIR"] = str(layout.dataset_dir)
    env["XGATE_RUNS_ROOT"] = str(layout.runs_root)
    return env


def expected_dataset_files(layout: ColabLayout) -> List[Path]:
    return [
        layout.dataset_dir / "final" / "final_balanced_train.parquet",
        layout.dataset_dir / "checkpoint3_val.parquet",
        layout.dataset_dir / "checkpoint3_test.parquet",
    ]


def missing_dataset_files(layout: ColabLayout) -> List[Path]:
    return [path for path in expected_dataset_files(layout) if not path.exists()]


def summarize_layout(layout: ColabLayout) -> str:
    return "\n".join(
        [
            f"Drive root         : {layout.drive_root}",
            f"Code directory     : {layout.code_dir}",
            f"Dataset directory  : {layout.dataset_dir}",
            f"Runs root          : {layout.runs_root}",
            f"Canonical results  : {layout.canonical_root}",
            f"Ablation results   : {layout.ablation_root}",
            f"Claim closure root : {layout.claim_closure_root}",
        ]
    )
