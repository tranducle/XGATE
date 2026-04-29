"""
================================================================================
X-GATE Claim Closure / Supplemental Evaluation Script
================================================================================
Consumes completed canonical checkpoints and computes the evidence that the
main multi-run script does not currently produce:

  - Float test metrics from saved checkpoints
  - Logical Drift (Delta_L) on held-out test data
  - Explanation-evasion adversarial metrics (Adv-F1 / Adv-FPR)
  - CPU-side deployment latency
  - Dynamic INT8 post-training deployment metrics and artifact size
  - Parameter counts and compression ratios

This script is intended to be run AFTER run_all_experiments.py has produced
checkpoints under a canonical results directory.
================================================================================
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.training.data_loader import get_dataloaders, INPUT_FEATURES, NUM_CLASSES
from src.model.SecurityBERT_Model import VanillaSecurityBERT, TinySecurityBERT
from src.training.eval_logical_drift import evaluate_logical_drift
from src.training.xgate_core import (
    evaluate_adversarial_fpr,
    evaluate_classifier,
    evaluate_latency_ms,
    quantize_dynamic_int8,
    serialized_model_size_mb,
)
from run_all_experiments import NpEncoder, bootstrap_ci, setup_logger


SEEDS = [42, 7, 13]
DEFAULT_RESULTS_ROOT = Path(r"C:\Users\Tran Duc Le\Documents\RESEARCHAGENTFINAL\projects\XGATE\ablation-results-colab2")
DEFAULT_OUTPUT_ROOT = Path("results/claim_closure")
DEFAULT_BATCH_SIZE = 1024
DEFAULT_DRIFT_SAMPLES = 1000
DEFAULT_ADV_SAMPLES = 2000


MODEL_FACTORIES = {
    "Vanilla_SecurityBERT_Teacher": lambda: VanillaSecurityBERT(
        num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
    ),
    "Full_XGATE": lambda: TinySecurityBERT(
        num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
    ),
    "KD_only": lambda: TinySecurityBERT(
        num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
    ),
    "KD_ECD": lambda: TinySecurityBERT(
        num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
    ),
    "KD_EGAT": lambda: TinySecurityBERT(
        num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Supplemental claim-closure evaluation for X-GATE.")
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--n-runs", type=int, default=len(SEEDS))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-device", type=str, default=None, help="cuda or cpu for float-model evaluation")
    parser.add_argument("--drift-samples", type=int, default=DEFAULT_DRIFT_SAMPLES)
    parser.add_argument("--adv-samples", type=int, default=DEFAULT_ADV_SAMPLES)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap for quicker dry-runs. When omitted, use the full held-out test split.",
    )
    return parser.parse_args()


def count_trainable_params_millions(model: torch.nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) / 1e6


def checkpoint_size_mb(checkpoint_path: Path) -> float:
    return checkpoint_path.stat().st_size / (1024 * 1024)


def maybe_limit_loader(dataloader, max_samples: Optional[int]):
    if max_samples is None:
        return dataloader

    collected_inputs = []
    collected_targets = []
    seen = 0
    for batch_inputs, batch_targets in dataloader:
        remaining = max_samples - seen
        if remaining <= 0:
            break
        if batch_inputs.size(0) > remaining:
            batch_inputs = batch_inputs[:remaining]
            batch_targets = batch_targets[:remaining]
        collected_inputs.append(batch_inputs)
        collected_targets.append(batch_targets)
        seen += batch_inputs.size(0)

    if not collected_inputs:
        return dataloader

    dataset = torch.utils.data.TensorDataset(
        torch.cat(collected_inputs, dim=0),
        torch.cat(collected_targets, dim=0),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def load_checkpoint_model(model_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = MODEL_FACTORIES[model_name]()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)


def evaluate_float_and_int8(
    model_name: str,
    model: torch.nn.Module,
    checkpoint_path: Path,
    eval_loader,
    eval_device: torch.device,
    cpu_device: torch.device,
    artifact_dir: Path,
) -> Dict[str, float]:
    logger = logging.getLogger("claim_closure")
    metrics = {}

    logger.info(f"[*] Evaluating Float {model_name}...")
    metrics["param_count_m"] = count_trainable_params_millions(model)
    metrics["float_ckpt_size_mb"] = checkpoint_size_mb(checkpoint_path)
    logger.info(f"[*] Measuring Float CPU Latency for {model_name}...")
    metrics["float_latency_cpu_ms"] = evaluate_latency_ms(model.cpu(), cpu_device, INPUT_FEATURES)["latency_ms_mean"]

    model = model.to(eval_device)
    logger.info(f"[*] Evaluating Float Test Set for {model_name} on {eval_device}...")
    float_test = evaluate_classifier(model, eval_loader, eval_device, NUM_CLASSES)
    for key, value in float_test.items():
        metrics[f"float_{key}"] = float(value)

    logger.info(f"[*] Quantizing {model_name} to INT8...")
    quantized_model = quantize_dynamic_int8(model)
    logger.info(f"[*] Evaluating INT8 Test Set for {model_name} on CPU...")
    int8_test = evaluate_classifier(quantized_model, eval_loader, cpu_device, NUM_CLASSES)
    for key, value in int8_test.items():
        metrics[f"int8_{key}"] = float(value)

    metrics["int8_latency_cpu_ms"] = evaluate_latency_ms(
        quantized_model, cpu_device, INPUT_FEATURES
    )["latency_ms_mean"]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    int8_artifact = artifact_dir / f"{model_name}_dynamic_int8_state.pth"
    metrics["int8_ckpt_size_mb"] = serialized_model_size_mb(quantized_model, int8_artifact)
    metrics["int8_size_ratio"] = metrics["int8_ckpt_size_mb"] / max(metrics["float_ckpt_size_mb"], 1e-9)
    return metrics


def aggregate_runs(run_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    model_names = sorted({model_name for run in run_results for model_name in run.keys()})

    for model_name in model_names:
        aggregated[model_name] = {}
        metric_names = sorted(
            {
                metric_name
                for run in run_results
                if model_name in run
                for metric_name in run[model_name].keys()
            }
        )
        for metric_name in metric_names:
            values = [
                run[model_name][metric_name]
                for run in run_results
                if model_name in run and metric_name in run[model_name]
            ]
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            ci_lo, ci_hi = bootstrap_ci(arr) if len(arr) > 1 else (arr[0], arr[0])
            aggregated[model_name][metric_name] = {
                "values": arr.tolist(),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "ci95_lo": float(ci_lo),
                "ci95_hi": float(ci_hi),
                "n": int(len(arr)),
            }
    return aggregated


def build_summary_table(stats: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    columns = [
        ("float_f1_macro", "Float F1"),
        ("int8_f1_macro", "INT8 F1"),
        ("adv_fpr_macro", "Adv FPR"),
        ("delta_L", "Delta_L"),
        ("float_latency_cpu_ms", "CPU Lat"),
        ("int8_size_ratio", "INT8 Size"),
    ]

    name_width = 32
    col_width = 14
    sep = "-" * (name_width + col_width * len(columns))
    lines = [sep]
    header = f"{'Model':<{name_width}}" + "".join(label.center(col_width) for _, label in columns)
    lines.append(header)
    lines.append(sep)

    for model_name, metrics in stats.items():
        row = f"{model_name:<{name_width}}"
        for metric_name, _ in columns:
            metric = metrics.get(metric_name)
            if not metric:
                row += "N/A".center(col_width)
                continue

            mean = metric["mean"]
            std = metric["std"]
            if "f1" in metric_name or "fpr" in metric_name:
                cell = f"{mean*100:.2f}±{std*100:.2f}%"
            elif "latency" in metric_name:
                cell = f"{mean:.4f}ms"
            else:
                cell = f"{mean:.4f}±{std:.4f}"
            row += cell.center(col_width)
        lines.append(row)

    lines.append(sep)
    lines.append("Float/INT8 metrics are held-out test results from saved checkpoints.")
    lines.append("Adv FPR and Delta_L are reported only for models where a teacher/student comparison is meaningful.")
    return "\n".join(lines)


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "claim_closure.log")

    eval_device = torch.device(args.eval_device) if args.eval_device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cpu_device = torch.device("cpu")

    logger.info("=" * 60)
    logger.info(" X-GATE Claim Closure Evaluation")
    logger.info("=" * 60)
    logger.info(f" Results root    : {results_root}")
    logger.info(f" Output root     : {output_root}")
    logger.info(f" Eval device     : {eval_device}")
    logger.info(f" Batch size      : {args.batch_size}")
    logger.info(f" Drift samples   : {args.drift_samples}")
    logger.info(f" Adv samples     : {args.adv_samples}")
    logger.info(f" Max eval sample : {args.max_eval_samples}")
    logger.info("=" * 60)

    warnings.filterwarnings("ignore")
    _, _, test_loader, meta = get_dataloaders(batch_size=args.batch_size, include_test_loader=True)
    eval_loader = maybe_limit_loader(test_loader, args.max_eval_samples)
    effective_samples = len(eval_loader.dataset) if hasattr(eval_loader, "dataset") else meta["test_samples"]
    logger.info(
        f" Test samples    : {meta['test_samples']:,} "
        f"(effective={effective_samples:,})"
    )
    logger.info(f" Dataset dir     : {meta['dataset_dir']}")

    all_run_results: List[Dict[str, Dict[str, float]]] = []

    for run_index in range(min(args.n_runs, len(SEEDS))):
        seed = SEEDS[run_index]
        run_dir = results_root / f"run_{run_index + 1:02d}_seed{seed}"
        if not run_dir.exists():
            logger.warning(f"Skipping missing run directory: {run_dir}")
            continue

        logger.info(f"\n{'#' * 60}")
        logger.info(f"# RUN {run_index + 1}/{args.n_runs} (seed={seed})")
        logger.info(f"{'#' * 60}")

        run_metrics: Dict[str, Dict[str, float]] = {}
        # Find the directory for this seed flexibly
        candidate_dirs = list(results_root.glob(f"*_seed{seed}"))
        if not candidate_dirs:
            logger.warning(f"No results directory found for seed {seed} in {results_root}")
            continue

        # Prefer the one with more files if multiple exist
        run_dir = max(candidate_dirs, key=lambda d: len(list(d.glob("*"))))
        logger.info(f"Using results directory: {run_dir}")

        run_artifacts_dir = output_root / f"run_{run_index + 1:02d}_seed{seed}"
        # Unified teacher path for all runs
        teacher_path = results_root / "teacher_root" / "run_03_seed13" / "Vanilla_SecurityBERT_Teacher_best.pth"
        if not teacher_path.exists():
            # Fallback if I moved it elsewhere during session
            logger.warning(f"Teacher path {teacher_path} not found.")
        if not teacher_path.exists():
            logger.warning(f"Teacher checkpoint not ready yet: {teacher_path}")
            continue

        teacher = load_checkpoint_model("Vanilla_SecurityBERT_Teacher", teacher_path, eval_device)
        run_metrics["Vanilla_SecurityBERT_Teacher"] = evaluate_float_and_int8(
            "Vanilla_SecurityBERT_Teacher",
            teacher,
            teacher_path,
            eval_loader,
            eval_device,
            cpu_device,
            run_artifacts_dir,
        )

        for student_name in ("Full_XGATE", "KD_only", "KD_ECD", "KD_EGAT"):
            checkpoint_path = run_dir / f"{student_name}_best.pth"
            if not checkpoint_path.exists():
                logger.info(f"Checkpoint not ready yet: {checkpoint_path}")
                continue

            student = load_checkpoint_model(student_name, checkpoint_path, eval_device)
            student_metrics = evaluate_float_and_int8(
                student_name,
                student,
                checkpoint_path,
                eval_loader,
                eval_device,
                cpu_device,
                run_artifacts_dir,
            )

            adv = evaluate_adversarial_fpr(
                student,
                teacher,
                eval_loader,
                eval_device,
                num_classes=NUM_CLASSES,
                epsilon=0.03,
                num_samples=args.adv_samples,
            )
            for key, value in adv.items():
                student_metrics[key] = float(value)

            spearman_rho, delta_l = evaluate_logical_drift(
                teacher,
                student,
                eval_loader,
                eval_device,
                num_samples=args.drift_samples,
            )
            student_metrics["spearman_rho"] = float(spearman_rho)
            student_metrics["delta_L"] = float(delta_l)
            run_metrics[student_name] = student_metrics

        if run_metrics:
            all_run_results.append(run_metrics)
            with open(run_artifacts_dir / "claim_closure_metrics.json", "w", encoding="utf-8") as handle:
                json.dump(run_metrics, handle, indent=2, cls=NpEncoder)

    if not all_run_results:
        logger.error("No completed checkpoints were available for claim-closure evaluation.")
        return

    aggregated = aggregate_runs(all_run_results)
    stats_path = output_root / "CLAIM_CLOSURE_STATS.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(aggregated, handle, indent=2, cls=NpEncoder)
    logger.info(f"\nAggregated stats -> {stats_path}")

    table = build_summary_table(aggregated)
    table_path = output_root / "CLAIM_CLOSURE_TABLE.txt"
    table_path.write_text(table, encoding="utf-8")
    logger.info("\n" + table)
    logger.info(f"Summary table    -> {table_path}")
    logger.info("\nClaim-closure evaluation complete.")


if __name__ == "__main__":
    main()
