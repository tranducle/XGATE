"""
================================================================================
X-GATE Unified Multi-Run Training & Statistical Analysis Script
================================================================================
Runs ALL models (Teacher, Student baselines, X-GATE) across N_RUNS seeds.
After all runs complete, computes mean ± std and 95% bootstrap CI for every metric.

Run from the XGATE/ project root:
    python run_all_experiments.py

Outputs:
    results/multirun/  <-- per-run checkpoints & JSON logs
    results/multirun/FINAL_STATS.json       <-- aggregated mean/std/CI table
    results/multirun/FINAL_STATS_TABLE.txt  <-- human-readable ASCII table
================================================================================
"""

import argparse
import os
import sys
import json
import time
import random
import logging
import warnings
import traceback
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── add src/ to path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.training.data_loader import (
    get_dataloaders,
    INPUT_FEATURES,
    NUM_CLASSES,
)
from src.model.SecurityBERT_Model import VanillaSecurityBERT, TinySecurityBERT
from src.model.baselines import (
    CNN1D_BiLSTM,
    MBConv_ViT_1D,
    TBCLNN,
    MLBaselineWrapper,
)
from src.training.Trainer import ExperimentTrainer, NpEncoder
from src.training.eval_logical_drift import evaluate_logical_drift
from src.training.xgate_core import (
    XGateLossConfig,
    evaluate_classifier,
    train_xgate_variant,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit here if needed)
# ─────────────────────────────────────────────────────────────────────────────

N_RUNS      = 5          # Number of independent random-seed runs
SEEDS       = [42, 7, 13, 99, 2025]   # One seed per run

# Epochs per model (ML baselines train in 1 shot, no epochs needed)
TEACHER_EPOCHS   = 10    # Vanilla SecurityBERT Teacher
STUDENT_EPOCHS   = 8     # Tiny Student models (X-GATE & Standard KD)
DL_EPOCHS        = 8     # CNN1D-BiLSTM, MBConv-ViT, TBCLNN

LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 1e-4
KD_ALPHA         = 0.5   # Weight for CE loss in Standard KD
KD_TEMP          = 4.0   # Softmax temperature for KD

# Logical Drift evaluation sample count
DRIFT_SAMPLES    = 1000

RESULTS_ROOT     = Path("results/multirun")
LOG_FILE         = RESULTS_ROOT / "run_all.log"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("XGATE_MULTIRUN")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD KD TRAINING  (has its own loss formula, not via ExperimentTrainer)
# ─────────────────────────────────────────────────────────────────────────────

def train_standard_kd(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    val_loader,
    device,
    num_epochs: int,
    run_dir: Path,
    logger: logging.Logger,
    alpha: float = KD_ALPHA,
    temp: float = KD_TEMP,
) -> Dict:
    """Train Standard Vanilla KD student and return best val metrics."""
    student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(),
                                   lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    criterion_ce  = nn.CrossEntropyLoss()
    criterion_kd  = nn.KLDivLoss(reduction="batchmean")

    best_f1    = 0.0
    best_state = None
    ckpt_path  = run_dir / "Standard_KD_TinyStudent_best.pth"

    from sklearn.metrics import (
        precision_recall_fscore_support, roc_auc_score,
        accuracy_score, confusion_matrix
    )

    def _evaluate(loader) -> Dict:
        student.eval()
        all_preds, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = student(X)
                probs  = torch.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        acc    = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        FP  = cm.sum(axis=0) - np.diag(cm)
        TN  = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
        fpr = float(np.mean(FP / (FP + TN + 1e-9)))
        return {"accuracy": acc, "precision_macro": p, "recall_macro": r,
                "f1_macro": f1, "roc_auc_macro": auc, "fpr_macro": fpr}

    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                logits_T = teacher(X)
            optimizer.zero_grad()
            logits_S = student(X)
            loss_ce  = criterion_ce(logits_S, y)
            loss_kd  = criterion_kd(
                F.log_softmax(logits_S / temp, dim=1),
                F.softmax(logits_T / temp, dim=1)
            ) * (temp * temp)
            loss = alpha * loss_ce + (1 - alpha) * loss_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        metrics = _evaluate(val_loader)
        logger.info(f"  [STD-KD] Epoch {epoch}/{num_epochs} | "
                    f"Loss={total_loss/len(train_loader):.4f} | "
                    f"F1={metrics['f1_macro']:.4f} | "
                    f"AUC={metrics['roc_auc_macro']:.4f}")
        if metrics["f1_macro"] > best_f1:
            best_f1    = metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    if best_state:
        student.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)
    return {"best_f1_macro": best_f1, "val_metrics": _evaluate(val_loader)}

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE RUN  (all models, one seed)
# ─────────────────────────────────────────────────────────────────────────────

def _reload_best_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model.to(device)


def _try_reuse_completed_teacher(
    reuse_root: Optional[Path],
    run_idx: int,
    seed: int,
    current_run_dir: Path,
    logger: logging.Logger,
) -> Optional[Dict]:
    if reuse_root is None:
        return None

    source_run_dir = reuse_root / f"run_{run_idx:02d}_seed{seed}"
    source_ckpt = source_run_dir / "Vanilla_SecurityBERT_Teacher_best.pth"
    source_history = source_run_dir / "Vanilla_SecurityBERT_Teacher_history.json"
    if not (source_ckpt.exists() and source_history.exists()):
        return None

    current_run_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = current_run_dir / source_ckpt.name
    target_history = current_run_dir / source_history.name
    shutil.copy2(source_ckpt, target_ckpt)
    shutil.copy2(source_history, target_history)

    with open(source_history, "r", encoding="utf-8") as handle:
        history = json.load(handle)

    logger.info(
        f"  Reusing completed Teacher from {source_run_dir} "
        f"(best val F1={history.get('best_f1_macro', 0.0):.4f})"
    )
    return history


def _fit_ml_baseline(model, train_loader):
    features = []
    targets = []
    for batch_inputs, batch_targets in train_loader:
        features.append(batch_inputs.numpy())
        targets.append(batch_targets.numpy())
    model.fit(np.vstack(features), np.concatenate(targets))


def run_single_seed(
    seed: int,
    run_idx: int,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    logger: logging.Logger,
    reuse_teacher_root: Optional[Path] = None,
) -> Dict:
    """Train every model once for the given seed and report held-out test metrics."""
    set_seed(seed)
    run_dir = RESULTS_ROOT / f"run_{run_idx:02d}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_all: Dict[str, Dict] = {}

    logger.info(f"\n{'='*60}")
    logger.info(f" [Run {run_idx}] Training Teacher: VanillaSecurityBERT")
    logger.info(f"{'='*60}")
    teacher = VanillaSecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES)
    teacher_result = _try_reuse_completed_teacher(reuse_teacher_root, run_idx, seed, run_dir, logger)
    if teacher_result is None:
        teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        teacher_trainer = ExperimentTrainer(
            model=teacher,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=teacher_optimizer,
            model_name="Vanilla_SecurityBERT_Teacher",
            num_classes=NUM_CLASSES,
            results_dir=str(run_dir),
            num_epochs=TEACHER_EPOCHS,
        )
        teacher_result = teacher_trainer.train(num_epochs=TEACHER_EPOCHS)
    teacher = _reload_best_checkpoint(teacher, run_dir / "Vanilla_SecurityBERT_Teacher_best.pth", device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    teacher_metrics = evaluate_classifier(teacher, test_loader, device, NUM_CLASSES)
    metrics_all["Vanilla_SecurityBERT_Teacher"] = teacher_metrics
    logger.info(
        f"  >>> Teacher test F1: {teacher_metrics['f1_macro']:.4f} "
        f"(best val F1={teacher_result.get('best_f1_macro', 0.0):.4f})"
    )

    xgate_loss = XGateLossConfig(
        ce_weight=1.0,
        kd_weight=1.0,
        fidelity_weight=0.5,
        adversarial_weight=0.3,
        kd_temp=KD_TEMP,
        epsilon=0.03,
        tau=1.0,
        validate_every=1,
    )
    logger.info(f"\n[Run {run_idx}] Training X-GATE TinyStudent")
    xgate = TinySecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES)
    xgate_result = train_xgate_variant(
        teacher=teacher,
        student=xgate,
        train_loader=train_loader,
        selection_loader=val_loader,
        report_loader=test_loader,
        device=device,
        num_classes=NUM_CLASSES,
        config_name="X-GATE_TinyStudent",
        loss_config=xgate_loss,
        num_epochs=STUDENT_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        run_dir=run_dir,
        logger=logger,
    )
    xgate = _reload_best_checkpoint(xgate, run_dir / "X-GATE_TinyStudent_best.pth", device)
    metrics_all["X-GATE_TinyStudent"] = xgate_result["report_metrics"]
    logger.info(
        f"  >>> X-GATE test F1: {xgate_result['report_metrics']['f1_macro']:.4f} "
        f"(best val F1={xgate_result['best_selection_f1_macro']:.4f})"
    )

    logger.info(f"\n[Run {run_idx}] Training Standard KD TinyStudent")
    standard_kd = TinySecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES)
    standard_kd_result = train_standard_kd(
        teacher=teacher,
        student=standard_kd,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=STUDENT_EPOCHS,
        run_dir=run_dir,
        logger=logger,
        alpha=KD_ALPHA,
        temp=KD_TEMP,
    )
    standard_kd = _reload_best_checkpoint(standard_kd, run_dir / "Standard_KD_TinyStudent_best.pth", device)
    metrics_all["Standard_KD_TinyStudent"] = evaluate_classifier(standard_kd, test_loader, device, NUM_CLASSES)
    logger.info(
        f"  >>> Standard KD test F1: {metrics_all['Standard_KD_TinyStudent']['f1_macro']:.4f} "
        f"(best val F1={standard_kd_result['best_f1_macro']:.4f})"
    )

    logger.info(f"\n[Run {run_idx}] Evaluating Logical Drift on held-out test split")
    spearman_x, drift_x = evaluate_logical_drift(teacher, xgate, test_loader, device, num_samples=DRIFT_SAMPLES)
    spearman_kd, drift_kd = evaluate_logical_drift(teacher, standard_kd, test_loader, device, num_samples=DRIFT_SAMPLES)
    metrics_all["X-GATE_TinyStudent"]["delta_L"] = float(drift_x)
    metrics_all["Standard_KD_TinyStudent"]["delta_L"] = float(drift_kd)
    logger.info(f"  X-GATE delta_L: {drift_x:.4f} (rho={spearman_x:.4f})")
    logger.info(f"  Standard KD delta_L: {drift_kd:.4f} (rho={spearman_kd:.4f})")

    for model_name, model_factory in [
        ("CNN1D_BiLSTM", lambda: CNN1D_BiLSTM(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)),
        ("MBConv_ViT", lambda: MBConv_ViT_1D(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)),
        ("TBCLNN", lambda: TBCLNN(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)),
    ]:
        logger.info(f"\n[Run {run_idx}] Training {model_name}")
        model = model_factory()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        trainer = ExperimentTrainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            model_name=model_name,
            num_classes=NUM_CLASSES,
            results_dir=str(run_dir),
            num_epochs=DL_EPOCHS,
        )
        trainer_result = trainer.train(num_epochs=DL_EPOCHS)
        model = _reload_best_checkpoint(model, run_dir / f"{model_name}_best.pth", device)
        metrics_all[model_name] = evaluate_classifier(model, test_loader, device, NUM_CLASSES)
        logger.info(
            f"  >>> {model_name} test F1: {metrics_all[model_name]['f1_macro']:.4f} "
            f"(best val F1={trainer_result.get('best_f1_macro', 0.0):.4f})"
        )

    logger.info(f"\n[Run {run_idx}] Training ML Baselines (LightGBM & RandomForest)")
    for model_name, model_type in [("LightGBM", "lightgbm"), ("RandomForest", "rf")]:
        baseline = MLBaselineWrapper(model_type=model_type, random_state=seed)
        _fit_ml_baseline(baseline, train_loader)
        metrics_all[model_name] = evaluate_classifier(baseline, test_loader, device, NUM_CLASSES)
        logger.info(f"  >>> {model_name} test F1: {metrics_all[model_name]['f1_macro']:.4f}")

    run_json = run_dir / "run_metrics.json"
    with open(run_json, "w", encoding="utf-8") as handle:
        json.dump({"seed": seed, "run": run_idx, "metrics": metrics_all}, handle, indent=2, cls=NpEncoder)
    logger.info(f"\n[Run {run_idx}] Saved -> {run_json}")

    return metrics_all

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CI  (on final per-run values, 10k resamples)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                 ci: float = 0.95) -> tuple:
    """
    Returns (lower_bound, upper_bound) for the given 95% CI
    using the percentile bootstrap method.
    """
    rng   = np.random.default_rng(seed=0)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, 100 * alpha)), \
           float(np.percentile(boots, 100 * (1 - alpha)))

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_runs(all_run_metrics: List[Dict]) -> Dict:
    """
    Aggregates results across N runs:
     - mean, std, 95% bootstrap CI for each (model, metric) pair.
    """
    METRICS_OF_INTEREST = [
        "f1_macro", "roc_auc_macro", "accuracy",
        "precision_macro", "recall_macro", "fpr_macro", "delta_L"
    ]

    # Collect all model names present across all runs
    all_models = set()
    for run in all_run_metrics:
        all_models.update(run.keys())

    aggregated = {}
    for model in sorted(all_models):
        aggregated[model] = {}
        for metric in METRICS_OF_INTEREST:
            vals = [run[model][metric]
                    for run in all_run_metrics
                    if model in run and metric in run[model]]
            if not vals:
                continue
            arr  = np.array(vals, dtype=float)
            lo, hi = bootstrap_ci(arr)
            aggregated[model][metric] = {
                "values": arr.tolist(),
                "mean":   float(arr.mean()),
                "std":    float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n":      len(arr),
            }
    return aggregated

# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_stats_table(stats: Dict, logger: logging.Logger) -> str:
    COLS    = ["f1_macro", "roc_auc_macro", "fpr_macro", "delta_L"]
    COL_W   = 28
    NAME_W  = 38
    sep     = "-" * (NAME_W + COL_W * len(COLS))

    header  = f"{'Model':<{NAME_W}}"
    for c in COLS:
        header += f"{c:^{COL_W}}"
    lines   = [sep, header, sep]

    for model, mdict in stats.items():
        row = f"{model:<{NAME_W}}"
        for c in COLS:
            if c in mdict:
                d  = mdict[c]
                s  = f"{d['mean']*100:.2f}±{d['std']*100:.2f}%"
                row += f"{s:^{COL_W}}"
            else:
                row += f"{'N/A':^{COL_W}}"
        lines.append(row)

    lines.append(sep)
    lines.append("  Format: mean ± std (%)  |  All F1/AUC/FPR reported as %;  delta_L is unitless")
    lines.append("  95% Bootstrap CI saved in FINAL_STATS.json")
    table = "\n".join(lines)
    logger.info("\n\n" + table)
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Canonical X-GATE multi-run experiment driver.")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--teacher-epochs", type=int, default=TEACHER_EPOCHS)
    parser.add_argument("--student-epochs", type=int, default=STUDENT_EPOCHS)
    parser.add_argument("--dl-epochs", type=int, default=DL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drift-samples", type=int, default=DRIFT_SAMPLES)
    parser.add_argument("--results-root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument(
        "--reuse-teacher-from",
        type=str,
        default=None,
        help="Optional root directory containing completed teacher checkpoints/history to reuse per run.",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--smoke-test", action="store_true", help="Run a minimal 1-seed sanity check.")
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global N_RUNS, TEACHER_EPOCHS, STUDENT_EPOCHS, DL_EPOCHS, DRIFT_SAMPLES, RESULTS_ROOT, LOG_FILE

    args = parse_args()
    if args.smoke_test:
        args.n_runs = 1
        args.teacher_epochs = 1
        args.student_epochs = 1
        args.dl_epochs = 1
        args.drift_samples = min(args.drift_samples, 64)

    N_RUNS = args.n_runs
    TEACHER_EPOCHS = args.teacher_epochs
    STUDENT_EPOCHS = args.student_epochs
    DL_EPOCHS = args.dl_epochs
    DRIFT_SAMPLES = args.drift_samples
    RESULTS_ROOT = Path(args.results_root)
    LOG_FILE = RESULTS_ROOT / "run_all.log"
    reuse_teacher_root = Path(args.reuse_teacher_from) if args.reuse_teacher_from else None

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOG_FILE)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info(" X-GATE Multi-Run Unified Training Script")
    logger.info("=" * 60)
    logger.info(f" Device         : {device}")
    if torch.cuda.is_available():
        logger.info(f" GPU            : {torch.cuda.get_device_name(0)}")
    logger.info(f" N runs         : {N_RUNS}")
    logger.info(f" Seeds          : {SEEDS[:N_RUNS]}")
    logger.info(f" Teacher epochs : {TEACHER_EPOCHS}")
    logger.info(f" Student epochs : {STUDENT_EPOCHS}")
    logger.info(f" DL epochs      : {DL_EPOCHS}")
    logger.info(f" Batch size     : {args.batch_size}")
    logger.info(f" Drift samples  : {DRIFT_SAMPLES}")
    logger.info(f" Smoke test     : {args.smoke_test}")
    logger.info(f" Results root   : {RESULTS_ROOT}")
    logger.info(f" Reuse Teacher  : {reuse_teacher_root if reuse_teacher_root else 'None'}")
    logger.info("=" * 60)

    # ── Load data ONCE (shared across all runs; randomisation is inside models)
    logger.info("\n>>> Loading data (shared across all runs)...")
    warnings.filterwarnings("ignore")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        batch_size=args.batch_size,
        include_test_loader=True,
    )
    logger.info(
        f"    Train: {meta['train_samples']:,} | Val: {meta['val_samples']:,} | "
        f"Test: {meta['test_samples']:,} | Features: {meta['input_features']}"
    )
    logger.info(f"    Dataset dir  : {meta['dataset_dir']}")

    # ── Multi-run loop ─────────────────────────────────────────────────────────
    all_run_metrics: List[Dict] = []
    total_start = time.time()

    for i in range(N_RUNS):
        seed = SEEDS[i]
        logger.info(f"\n{'#'*60}")
        logger.info(f"# RUN {i+1}/{N_RUNS}  (seed={seed})")
        logger.info(f"{'#'*60}")
        t0 = time.time()
        try:
            run_metrics = run_single_seed(
                seed=seed, run_idx=i + 1,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                logger=logger,
                reuse_teacher_root=reuse_teacher_root,
            )
            all_run_metrics.append(run_metrics)
        except Exception as e:
            logger.error(f"Run {i+1} catastrophically failed: {e}")
            traceback.print_exc()
        elapsed = (time.time() - t0) / 60
        logger.info(f"  Run {i+1} finished in {elapsed:.1f} min")

    total_elapsed = (time.time() - total_start) / 60
    logger.info(f"\n{'='*60}")
    logger.info(f" All runs complete in {total_elapsed:.1f} min")
    logger.info(f"{'='*60}")

    if not all_run_metrics:
        logger.error("No successful runs. Cannot compute statistics.")
        return

    # ── Aggregate stats ────────────────────────────────────────────────────────
    logger.info("\n>>> Computing statistical aggregates...")
    stats = aggregate_runs(all_run_metrics)

    # Save JSON
    stats_json = RESULTS_ROOT / "FINAL_STATS.json"
    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2, cls=NpEncoder)
    logger.info(f"    Stats saved → {stats_json}")

    # Print and save ASCII table
    table_str = print_stats_table(stats, logger)
    table_path = RESULTS_ROOT / "FINAL_STATS_TABLE.txt"
    table_path.write_text(table_str, encoding="utf-8")
    logger.info(f"    Table saved → {table_path}")

    logger.info("\n DONE. Check results/multirun/FINAL_STATS_TABLE.txt for the summary.")


if __name__ == "__main__":
    main()
