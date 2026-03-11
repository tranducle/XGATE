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

import os
import sys
import json
import time
import random
import logging
import warnings
import traceback
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

def run_single_seed(
    seed: int,
    run_idx: int,
    device: torch.device,
    train_loader,
    val_loader,
    logger: logging.Logger,
) -> Dict:
    """
    Train every model once for the given seed and return a dict of metrics.
    Returns: {model_name: {f1_macro, roc_auc_macro, fpr_macro, accuracy, ...}}
    """
    set_seed(seed)
    run_dir = RESULTS_ROOT / f"run_{run_idx:02d}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_all: Dict[str, Dict] = {}

    # ── 1. Vanilla SecurityBERT Teacher ──────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f" [Run {run_idx}] Training Teacher: VanillaSecurityBERT")
    logger.info(f"{'='*60}")
    teacher = VanillaSecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES)
    optimizer = torch.optim.AdamW(teacher.parameters(),
                                   lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trainer = ExperimentTrainer(
        model=teacher, device=device,
        train_loader=train_loader, val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(), optimizer=optimizer,
        model_name="Vanilla_SecurityBERT_Teacher",
        num_classes=NUM_CLASSES, results_dir=str(run_dir),
        num_epochs=TEACHER_EPOCHS
    )
    try:
        result = trainer.train(num_epochs=TEACHER_EPOCHS)
        best = result.get("history", [{}])[-1] if result.get("history") else {}
        # Reload best checkpoint for Teacher so downstream KD uses the best weights
        ckpt_teacher = run_dir / "Vanilla_SecurityBERT_Teacher_best.pth"
        if ckpt_teacher.exists():
            teacher.load_state_dict(
                torch.load(ckpt_teacher, map_location=device, weights_only=True))
        teacher.to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        metrics_all["Vanilla_SecurityBERT_Teacher"] = {
            "f1_macro":       result.get("best_f1_macro", 0.0),
            "roc_auc_macro":  best.get("roc_auc_macro", 0.0),
            "fpr_macro":      best.get("fpr_macro", 0.0),
            "accuracy":       best.get("accuracy", 0.0),
            "precision_macro": best.get("precision_macro", 0.0),
            "recall_macro":   best.get("recall_macro", 0.0),
        }
        logger.info(f"  >>> Teacher F1: {result.get('best_f1_macro', 0):.4f}")
    except Exception as e:
        logger.error(f"  Teacher FAILED: {e}"); traceback.print_exc()
        teacher = VanillaSecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES)
        teacher.to(device); teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # ── 2. X-GATE TinyStudent ─────────────────────────────────────────────────
    logger.info(f"\n[Run {run_idx}] Training X-GATE TinyStudent")
    xgate = TinySecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES)
    optimizer = torch.optim.AdamW(xgate.parameters(),
                                   lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trainer = ExperimentTrainer(
        model=xgate, device=device,
        train_loader=train_loader, val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(), optimizer=optimizer,
        model_name="X-GATE_TinyStudent",
        num_classes=NUM_CLASSES, results_dir=str(run_dir),
        num_epochs=STUDENT_EPOCHS
    )
    try:
        result = trainer.train(num_epochs=STUDENT_EPOCHS)
        best = result.get("history", [{}])[-1] if result.get("history") else {}
        xgate_ckpt = run_dir / "X-GATE_TinyStudent_best.pth"
        if xgate_ckpt.exists():
            xgate.load_state_dict(
                torch.load(xgate_ckpt, map_location=device, weights_only=True))
        metrics_all["X-GATE_TinyStudent"] = {
            "f1_macro":        result.get("best_f1_macro", 0.0),
            "roc_auc_macro":   best.get("roc_auc_macro", 0.0),
            "fpr_macro":       best.get("fpr_macro", 0.0),
            "accuracy":        best.get("accuracy", 0.0),
            "precision_macro": best.get("precision_macro", 0.0),
            "recall_macro":    best.get("recall_macro", 0.0),
        }
        logger.info(f"  >>> X-GATE F1: {result.get('best_f1_macro', 0):.4f}")
    except Exception as e:
        logger.error(f"  X-GATE FAILED: {e}"); traceback.print_exc()

    # ── 3. Standard KD Student ────────────────────────────────────────────────
    logger.info(f"\n[Run {run_idx}] Training Standard KD TinyStudent")
    std_student = TinySecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES)
    try:
        kd_result = train_standard_kd(
            teacher=teacher, student=std_student,
            train_loader=train_loader, val_loader=val_loader,
            device=device, num_epochs=STUDENT_EPOCHS,
            run_dir=run_dir, logger=logger,
        )
        vm = kd_result.get("val_metrics", {})
        metrics_all["Standard_KD_TinyStudent"] = {
            "f1_macro":        kd_result.get("best_f1_macro", vm.get("f1_macro", 0.0)),
            "roc_auc_macro":   vm.get("roc_auc_macro", 0.0),
            "fpr_macro":       vm.get("fpr_macro", 0.0),
            "accuracy":        vm.get("accuracy", 0.0),
            "precision_macro": vm.get("precision_macro", 0.0),
            "recall_macro":    vm.get("recall_macro", 0.0),
        }
        std_kd_ckpt = run_dir / "Standard_KD_TinyStudent_best.pth"
        if std_kd_ckpt.exists():
            std_student.load_state_dict(
                torch.load(std_kd_ckpt, map_location=device, weights_only=True))
        logger.info(f"  >>> Standard KD F1: {kd_result.get('best_f1_macro', 0):.4f}")
    except Exception as e:
        logger.error(f"  Standard KD FAILED: {e}"); traceback.print_exc()

    # ── 4. Logical Drift: X-GATE vs Standard KD ───────────────────────────────
    logger.info(f"\n[Run {run_idx}] Evaluating Logical Drift")
    try:
        xgate.to(device); std_student.to(device)
        spearman_x, drift_x = evaluate_logical_drift(
            teacher, xgate, val_loader, device, num_samples=DRIFT_SAMPLES)
        if "X-GATE_TinyStudent" in metrics_all:
            metrics_all["X-GATE_TinyStudent"]["delta_L"] = float(drift_x)
        logger.info(f"  X-GATE Delta_L: {drift_x:.4f}")

        spearman_s, drift_s = evaluate_logical_drift(
            teacher, std_student, val_loader, device, num_samples=DRIFT_SAMPLES)
        if "Standard_KD_TinyStudent" in metrics_all:
            metrics_all["Standard_KD_TinyStudent"]["delta_L"] = float(drift_s)
        logger.info(f"  Standard KD Delta_L: {drift_s:.4f}")
    except Exception as e:
        logger.warning(f"  Logical Drift evaluation failed: {e}")

    # ── 5. CNN1D-BiLSTM ───────────────────────────────────────────────────────
    logger.info(f"\n[Run {run_idx}] Training CNN1D_BiLSTM")
    cnn = CNN1D_BiLSTM(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
    opt = torch.optim.AdamW(cnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    t   = ExperimentTrainer(
        model=cnn, device=device,
        train_loader=train_loader, val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(), optimizer=opt,
        model_name="CNN1D_BiLSTM", num_classes=NUM_CLASSES,
        results_dir=str(run_dir), num_epochs=DL_EPOCHS
    )
    try:
        r = t.train(num_epochs=DL_EPOCHS)
        b = (r.get("history") or [{}])[-1]
        metrics_all["CNN1D_BiLSTM"] = {
            "f1_macro":        r.get("best_f1_macro", 0.0),
            "roc_auc_macro":   b.get("roc_auc_macro", 0.0),
            "fpr_macro":       b.get("fpr_macro", 0.0),
            "accuracy":        b.get("accuracy", 0.0),
            "precision_macro": b.get("precision_macro", 0.0),
            "recall_macro":    b.get("recall_macro", 0.0),
        }
        logger.info(f"  >>> CNN1D_BiLSTM F1: {r.get('best_f1_macro', 0):.4f}")
    except Exception as e:
        logger.error(f"  CNN1D_BiLSTM FAILED: {e}"); traceback.print_exc()

    # ── 6. MBConv-ViT ─────────────────────────────────────────────────────────
    logger.info(f"\n[Run {run_idx}] Training MBConv_ViT")
    mbvit = MBConv_ViT_1D(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
    opt2  = torch.optim.AdamW(mbvit.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    t2    = ExperimentTrainer(
        model=mbvit, device=device,
        train_loader=train_loader, val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(), optimizer=opt2,
        model_name="MBConv_ViT", num_classes=NUM_CLASSES,
        results_dir=str(run_dir), num_epochs=DL_EPOCHS
    )
    try:
        r2 = t2.train(num_epochs=DL_EPOCHS)
        b2 = (r2.get("history") or [{}])[-1]
        metrics_all["MBConv_ViT"] = {
            "f1_macro":        r2.get("best_f1_macro", 0.0),
            "roc_auc_macro":   b2.get("roc_auc_macro", 0.0),
            "fpr_macro":       b2.get("fpr_macro", 0.0),
            "accuracy":        b2.get("accuracy", 0.0),
            "precision_macro": b2.get("precision_macro", 0.0),
            "recall_macro":    b2.get("recall_macro", 0.0),
        }
        logger.info(f"  >>> MBConv_ViT F1: {r2.get('best_f1_macro', 0):.4f}")
    except Exception as e:
        logger.error(f"  MBConv_ViT FAILED: {e}"); traceback.print_exc()

    # ── 7. TBCLNN ─────────────────────────────────────────────────────────────
    logger.info(f"\n[Run {run_idx}] Training TBCLNN")
    tbclnn = TBCLNN(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
    opt3   = torch.optim.AdamW(tbclnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    t3     = ExperimentTrainer(
        model=tbclnn, device=device,
        train_loader=train_loader, val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(), optimizer=opt3,
        model_name="TBCLNN", num_classes=NUM_CLASSES,
        results_dir=str(run_dir), num_epochs=DL_EPOCHS
    )
    try:
        r3 = t3.train(num_epochs=DL_EPOCHS)
        b3 = (r3.get("history") or [{}])[-1]
        metrics_all["TBCLNN"] = {
            "f1_macro":        r3.get("best_f1_macro", 0.0),
            "roc_auc_macro":   b3.get("roc_auc_macro", 0.0),
            "fpr_macro":       b3.get("fpr_macro", 0.0),
            "accuracy":        b3.get("accuracy", 0.0),
            "precision_macro": b3.get("precision_macro", 0.0),
            "recall_macro":    b3.get("recall_macro", 0.0),
        }
        logger.info(f"  >>> TBCLNN F1: {r3.get('best_f1_macro', 0):.4f}")
    except Exception as e:
        logger.error(f"  TBCLNN FAILED: {e}"); traceback.print_exc()

    # ── 8. Traditional ML Baselines (run on CPU; one fit, no epochs) ──────────
    logger.info(f"\n[Run {run_idx}] Training ML Baselines (LightGBM & RandomForest)")
    for ml_name, ml_type in [("LightGBM", "lightgbm"), ("RandomForest", "rf")]:
        try:
            ml_model = MLBaselineWrapper(model_type=ml_type)
            ml_trainer = ExperimentTrainer(
                model=ml_model, device=torch.device("cpu"),
                train_loader=train_loader, val_loader=val_loader,
                criterion=None, optimizer=None,
                model_name=ml_name, num_classes=NUM_CLASSES,
                results_dir=str(run_dir), num_epochs=1
            )
            ml_result = ml_trainer.train(num_epochs=1)
            vm = ml_result.get("metrics", {})
            metrics_all[ml_name] = {
                "f1_macro":        vm.get("f1_macro", 0.0),
                "roc_auc_macro":   vm.get("roc_auc_macro", 0.0),
                "fpr_macro":       vm.get("fpr_macro", 0.0),
                "accuracy":        vm.get("accuracy", 0.0),
                "precision_macro": vm.get("precision_macro", 0.0),
                "recall_macro":    vm.get("recall_macro", 0.0),
            }
            logger.info(f"  >>> {ml_name} F1: {vm.get('f1_macro', 0):.4f}")
        except Exception as e:
            logger.error(f"  {ml_name} FAILED: {e}"); traceback.print_exc()

    # ── Save this run's results ───────────────────────────────────────────────
    run_json = run_dir / "run_metrics.json"
    with open(run_json, "w") as f:
        json.dump({"seed": seed, "run": run_idx, "metrics": metrics_all},
                  f, indent=2, cls=NpEncoder)
    logger.info(f"\n[Run {run_idx}] Saved → {run_json}")

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

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOG_FILE)

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
    logger.info(f" Results root   : {RESULTS_ROOT}")
    logger.info("=" * 60)

    # ── Load data ONCE (shared across all runs; randomisation is inside models)
    logger.info("\n>>> Loading data (shared across all runs)...")
    warnings.filterwarnings("ignore")
    train_loader, val_loader, meta = get_dataloaders()
    logger.info(f"    Train: {meta['train_samples']:,} samples | "
                f"Val: {meta['eval_samples']:,} samples | "
                f"Features: {meta['input_features']}")

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
                logger=logger,
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
