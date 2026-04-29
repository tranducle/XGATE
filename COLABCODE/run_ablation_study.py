"""
================================================================================
X-GATE Ablation Study Script
================================================================================
Tests 4 ablation configurations to disentangle ECD and EGAT contributions:

  (a) Standard KD       — CE + KL divergence only (baseline, already run)
  (b) KD + ECD only     — CE + KL + Fidelity Loss (β > 0, γ = 0)
  (c) KD + EGAT only    — CE + KL + Adversarial Training (β = 0, γ > 0)
  (d) Full X-GATE       — CE + KL + Fidelity + Adversarial (β, γ > 0)

NOTE: Ready to run. Reuses existing Teacher checkpoints from results/multirun/.

Usage:
    python run_ablation_study.py

Prerequisites:
    - Trained Teacher checkpoints must exist in results/multirun/run_XX_seedYY/
    - Dataset pre-processed and available via get_dataloaders()

Outputs:
    results/ablation/ABLATION_STATS.json        — Per-run and aggregated results
    results/ablation/ABLATION_TABLE.txt          — Human-readable summary
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
import gc
from pathlib import Path
from typing import Dict, List
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - older torch versions
    SDPBackend = None
    sdpa_kernel = None

sys.path.insert(0, str(Path(__file__).parent))

from src.training.data_loader import get_dataloaders, INPUT_FEATURES, NUM_CLASSES
from src.model.SecurityBERT_Model import VanillaSecurityBERT, TinySecurityBERT
from src.training.eval_logical_drift import evaluate_logical_drift
from src.training.xgate_core import (
    XGateLossConfig,
    evaluate_adversarial_fpr as evaluate_adv_metrics,
    evaluate_latency_ms,
    train_xgate_variant as train_variant,
)
from run_all_experiments import (
    set_seed, setup_logger, bootstrap_ci, NpEncoder,
    LEARNING_RATE, WEIGHT_DECAY, KD_TEMP, DRIFT_SAMPLES,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

N_RUNS = 5
SEEDS  = [42, 7, 13, 99, 2025]
ABLATION_EPOCHS = 20  # Must match other models for fair comparison
BATCH_SIZE = 512      # MUST match original experiments (DEFAULT_BATCH_SIZE in data_loader.py)
VAL_EVERY  = 5        # Validate every N epochs (saves time)
ALPHA   = 1.0    # CE weight
BETA    = 0.5    # ECD fidelity weight
GAMMA   = 0.3    # EGAT adversarial weight
EPSILON = 0.03   # FGSM perturbation bound
TAU     = 1.0    # NeuralSort temperature

RESULTS_ROOT = Path("results/ablation")
TEACHER_ROOT = Path("results/multirun")  # Reuse existing Teacher checkpoints

# Ablation configurations
ABLATION_CONFIGS = {
    "KD_only":     {"beta": 0.0,  "gamma": 0.0},   # Standard KD (control)
    "KD_ECD":      {"beta": BETA, "gamma": 0.0},    # KD + ECD only
    "KD_EGAT":     {"beta": 0.0,  "gamma": GAMMA},  # KD + EGAT only
    "Full_XGATE":  {"beta": BETA, "gamma": GAMMA},  # Full X-GATE
}


# ─────────────────────────────────────────────────────────────────────────────
# SOFT-RANK SPEARMAN FIDELITY LOSS (ECD)
# ─────────────────────────────────────────────────────────────────────────────

def soft_rank(v: torch.Tensor, tau: float = TAU) -> torch.Tensor:
    """
    NeuralSort soft-ranking (Grover et al., 2019; Blondel et al., 2020).
    R_j = 1 + sum_{k != j} sigma((v_k - v_j) / tau)
    """
    d = v.shape[-1]
    # v: (batch, d)
    v_i = v.unsqueeze(-1)        # (batch, d, 1)
    v_j = v.unsqueeze(-2)        # (batch, 1, d)
    diff = (v_j - v_i) / tau     # (batch, d, d)
    pairwise = torch.sigmoid(diff)
    # Diagonal should not contribute
    mask = 1.0 - torch.eye(d, device=v.device).unsqueeze(0)
    ranks = 1.0 + (pairwise * mask).sum(dim=-1)  # (batch, d)
    return ranks


def fidelity_loss(phi_T: torch.Tensor, phi_S: torch.Tensor,
                  tau: float = TAU) -> torch.Tensor:
    """
    Soft-Rank Spearman Fidelity Loss (Eq. 4 in manuscript).
    L_fidelity = (6 * sum(d_i^2)) / (d * (d^2 - 1))
    where d_i = R(phi_T)_i - R(phi_S)_i
    Returns: scalar loss in [0, 1], where 0 = perfect alignment.
    """
    rank_T = soft_rank(phi_T.abs(), tau)
    rank_S = soft_rank(phi_S.abs(), tau)
    d = phi_T.shape[-1]
    d_sq = (rank_T - rank_S) ** 2
    loss = (6.0 * d_sq.sum(dim=-1)) / (d * (d ** 2 - 1))
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# INPUT × GRADIENT ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def input_x_gradient(model: nn.Module, x: torch.Tensor,
                     target: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """
    Computes Input × Gradient attributions (Eq. 2 in manuscript).
    Returns gradient-level tensor (batch, d).
    """
    x_attr = x.detach().clone().requires_grad_(True)
    if create_graph and x_attr.is_cuda and sdpa_kernel is not None and SDPBackend is not None:
        attention_ctx = sdpa_kernel(backends=[SDPBackend.MATH])
    else:
        attention_ctx = nullcontext()

    with attention_ctx:
        logits = model(x_attr)
    gathered = logits.gather(1, target.view(-1, 1)).squeeze(1)
    
    # Use autograd.grad to explicitly avoid accumulating gradients in model parameters!
    grads = torch.autograd.grad(
        outputs=gathered,
        inputs=x_attr,
        grad_outputs=torch.ones_like(gathered),
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True
    )[0]
    
    phi = grads * x_attr
    if not create_graph:
        phi = phi.detach()
    return phi


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL EVALUATION (Table IV: Explanation-Evasion FPR)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_adversarial_fpr(
    model: nn.Module, teacher: nn.Module, dataloader,
    device: torch.device, epsilon: float = EPSILON, num_samples: int = 2000,
) -> dict:
    """
    Evaluates a model's adversarial robustness under the Explanation-Evasion
    threat model described in §3.3 and Table IV of the manuscript.

    Attack protocol:
      1. Compute Teacher IxG attributions → semantic mask M
      2. FGSM perturbation: δ = ε · M ⊙ sign(∇_x L_CE)
      3. Measure student F1 and FPR on adversarial examples

    Returns dict with adv_f1_macro, adv_fpr_macro.
    """
    from sklearn.metrics import (
        precision_recall_fscore_support, confusion_matrix, accuracy_score,
    )

    model.eval()
    teacher.eval()
    all_preds, all_targets = [], []
    processed = 0

    for X, y in dataloader:
        if processed >= num_samples:
            break
        X, y = X.to(device), y.to(device)
        batch_size = X.size(0)

        # 1. Compute Teacher IxG → semantic mask
        phi_T = input_x_gradient(teacher, X, y)
        mask = phi_T.abs().detach()
        mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)

        # 2. FGSM semantic attack on the STUDENT model
        x_adv = fgsm_semantic_attack(model, X, y, mask, epsilon)

        # 3. Evaluate student on adversarial examples
        with torch.no_grad():
            logits_adv = model(x_adv)
            preds_adv = logits_adv.argmax(dim=1)

        all_preds.append(preds_adv.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        processed += batch_size

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    _, _, f1_adv, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    fpr_adv = float(np.mean(FP / (FP + TN + 1e-9)))

    return {"adv_f1_macro": float(f1_adv), "adv_fpr_macro": fpr_adv}


# ─────────────────────────────────────────────────────────────────────────────
# LATENCY PROFILING
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_latency(
    model: nn.Module, device: torch.device, input_dim: int = INPUT_FEATURES,
    n_warmup: int = 50, n_measure: int = 200,
) -> dict:
    """Measure inference latency in ms (single sample)."""
    import time as _time
    model.eval()
    dummy = torch.randn(1, input_dim, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = _time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((_time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(times)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50":  float(np.percentile(arr, 50)),
        "latency_ms_p99":  float(np.percentile(arr, 99)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FGSM SEMANTIC ATTACK (EGAT)
# ─────────────────────────────────────────────────────────────────────────────

def fgsm_semantic_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         mask: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    """
    FGSM with semantic importance mask (Eq. 7 in manuscript).
    delta = epsilon * M ⊙ sign(∇_x L_CE)

    NOTE: Data is StandardScaler-normalized (range approx [-5, 5]),
    so we clamp to [-5, 5] instead of [0, 1].
    """
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=x_adv,
        only_inputs=True
    )[0]
    
    grad_sign = grads.sign()
    delta = epsilon * mask * grad_sign
    x_pert = torch.clamp(x + delta.detach(), -5.0, 5.0)
    return x_pert.detach()


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_ablation_variant(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    config_name: str,
    beta: float,
    gamma: float,
    num_epochs: int,
    run_dir: Path,
    logger: logging.Logger,
) -> Dict:
    """
    Train a student with configurable ECD (beta) and EGAT (gamma) weights.

    Loss = alpha * L_CE + (1-alpha) * L_KD + beta * L_Fidelity
    If gamma > 0: additionally train on adversarial examples with semantic mask.
    """
    student.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # NOTE: No LR scheduler — matches original ExperimentTrainer and run_standard_kd.py
    ce_loss_fn = nn.CrossEntropyLoss()
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")

    # AMP: Disabled due to Infinity/NaN gradient issues in ECD/EGAT calculation
    # (The GradScaler was failing on `found_inf_per_device`)

    best_f1 = 0.0
    best_state = None
    ckpt_path = run_dir / f"{config_name}_best.pth"

    from sklearn.metrics import (
        precision_recall_fscore_support, roc_auc_score,
        accuracy_score, confusion_matrix,
    )

    def _evaluate(loader) -> Dict:
        student.eval()
        all_preds, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = student(X)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        FP = cm.sum(axis=0) - np.diag(cm)
        TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
        fpr = float(np.mean(FP / (FP + TN + 1e-9)))
        return {
            "accuracy": acc, "precision_macro": p, "recall_macro": r,
            "f1_macro": f1, "roc_auc_macro": auc, "fpr_macro": fpr,
        }

    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{config_name}] Epoch {epoch}/{num_epochs}")

        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            # ── Teacher forward (frozen) ──
            with torch.no_grad():
                logits_T = teacher(X)

            optimizer.zero_grad()

            # ── Student forward + Base losses (AMP float16 for speed) ──
            # ── Student forward + Base losses ──
            logits_S = student(X)
            loss_ce = ce_loss_fn(logits_S, y)
            loss_kd = kd_loss_fn(
                F.log_softmax(logits_S / KD_TEMP, dim=1),
                F.softmax(logits_T / KD_TEMP, dim=1),
            ) * (KD_TEMP ** 2)
            loss = KD_ALPHA * loss_ce + (1 - KD_ALPHA) * loss_kd

            # ── ECD: Fidelity Loss (if beta > 0) ──
            # Run in float32 for autograd.grad correctness
            if beta > 0:
                phi_T = input_x_gradient(teacher, X, y, create_graph=False)
                phi_S = input_x_gradient(student, X, y, create_graph=False)
                loss_fid = fidelity_loss(phi_T, phi_S, tau=TAU)

                loss_align = F.mse_loss(
                    logits_S.float().gather(1, y.view(-1, 1)),
                    logits_T.float().gather(1, y.view(-1, 1)).detach(),
                )
                loss = loss.float() + beta * (loss_fid.detach() * loss_align + loss_align)

            # ── EGAT: Adversarial Training (if gamma > 0) ──
            # Run in float32 for autograd.grad correctness
            if gamma > 0:
                if beta == 0:
                    phi_T_mask = input_x_gradient(teacher, X, y)
                else:
                    phi_T_mask = phi_T
                mask = phi_T_mask.abs().detach()
                mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)

                x_adv = fgsm_semantic_attack(student, X, y, mask, EPSILON)

                logits_S_adv = student(x_adv)
                loss_adv = F.cross_entropy(logits_S_adv, y)
                loss_kl_adv = F.kl_div(
                    F.log_softmax(logits_S_adv / KD_TEMP, dim=1),
                    F.softmax(logits_T / KD_TEMP, dim=1),
                    reduction="batchmean",
                ) * (KD_TEMP ** 2)

                loss = loss + gamma * (loss_adv + loss_kl_adv)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ── Validation (every VAL_EVERY epochs + last epoch) ──
        if epoch % VAL_EVERY == 0 or epoch == num_epochs:
            metrics = _evaluate(val_loader)
            logger.info(
                f"  [{config_name}] Epoch {epoch}/{num_epochs} | "
                f"Loss={total_loss/len(train_loader):.4f} | "
                f"F1={metrics['f1_macro']:.4f} | AUC={metrics['roc_auc_macro']:.4f}"
            )
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        else:
            logger.info(
                f"  [{config_name}] Epoch {epoch}/{num_epochs} | "
                f"Loss={total_loss/len(train_loader):.4f} (skip val)"
            )

    if best_state:
        student.load_state_dict(best_state)
        torch.save(best_state, ckpt_path)

    return {"best_f1_macro": best_f1, "val_metrics": _evaluate(val_loader)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Canonical X-GATE ablation driver.")
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--epochs", type=int, default=ABLATION_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--drift-samples", type=int, default=DRIFT_SAMPLES)
    parser.add_argument("--results-root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--teacher-root", type=str, default=str(TEACHER_ROOT))
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--smoke-test", action="store_true", help="Run a minimal 1-seed ablation sanity check.")
    return parser.parse_args()


def main():
    global N_RUNS, ABLATION_EPOCHS, BATCH_SIZE, DRIFT_SAMPLES, RESULTS_ROOT, TEACHER_ROOT

    args = parse_args()
    if args.smoke_test:
        args.n_runs = 1
        args.epochs = 1
        args.drift_samples = min(args.drift_samples, 64)

    N_RUNS = args.n_runs
    ABLATION_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DRIFT_SAMPLES = args.drift_samples
    RESULTS_ROOT = Path(args.results_root)
    TEACHER_ROOT = Path(args.teacher_root)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(RESULTS_ROOT / "ablation.log")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Speed optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels for fixed input size
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere GPUs (RTX 3090)

    logger.info("=" * 60)
    logger.info(" X-GATE Ablation Study")
    logger.info("=" * 60)
    logger.info(f" Device: {device}")
    logger.info(f" Configs: {list(ABLATION_CONFIGS.keys())}")
    logger.info(f" Runs: {N_RUNS}, Seeds: {SEEDS[:N_RUNS]}")
    logger.info(f" AMP: {'Yes' if device.type == 'cuda' else 'No'}, Batch: {BATCH_SIZE}")
    logger.info(f" Smoke test: {args.smoke_test}")
    logger.info(f" Teacher root: {TEACHER_ROOT}")
    logger.info("=" * 60)

    # Load data
    warnings.filterwarnings("ignore")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        batch_size=BATCH_SIZE,
        include_test_loader=True,
    )
    logger.info(
        f" Data: {meta['train_samples']:,} train, {meta['val_samples']:,} val, "
        f"{meta['test_samples']:,} test (batch={BATCH_SIZE})"
    )
    logger.info(f" Dataset dir: {meta['dataset_dir']}")

    all_results: List[Dict] = []

    for run_idx in range(N_RUNS):
        seed = SEEDS[run_idx]
        set_seed(seed)
        run_dir = RESULTS_ROOT / f"run_{run_idx+1:02d}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'#'*60}")
        logger.info(f"# RUN {run_idx+1}/{N_RUNS} (seed={seed})")
        logger.info(f"{'#'*60}")

        # ── Load pre-trained Teacher ──
        teacher_ckpt = TEACHER_ROOT / f"run_{run_idx+1:02d}_seed{seed}" / \
                       "Vanilla_SecurityBERT_Teacher_best.pth"
        teacher = VanillaSecurityBERT(
            num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
        ).to(device)

        if teacher_ckpt.exists():
            teacher.load_state_dict(
                torch.load(teacher_ckpt, map_location=device, weights_only=True)
            )
            logger.info(f"  Loaded Teacher from {teacher_ckpt}")
        else:
            logger.warning(f"  Teacher checkpoint not found: {teacher_ckpt}")
            logger.warning("  Training teacher from scratch...")
            # Optional: train teacher here, or skip run
            continue

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        run_metrics = {}

        for config_name, config in ABLATION_CONFIGS.items():
            logger.info(f"\n  ── {config_name} (β={config['beta']}, γ={config['gamma']})")
            set_seed(seed)  # Reset seed for fair comparison

            try:
                student = TinySecurityBERT(
                    num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
                )

                result = train_variant(
                    teacher=teacher,
                    student=student,
                    train_loader=train_loader,
                    selection_loader=val_loader,
                    report_loader=test_loader,
                    device=device,
                    num_classes=NUM_CLASSES,
                    config_name=config_name,
                    loss_config=XGateLossConfig(
                        ce_weight=1.0,
                        kd_weight=1.0,
                        fidelity_weight=config["beta"],
                        adversarial_weight=config["gamma"],
                        kd_temp=KD_TEMP,
                        epsilon=EPSILON,
                        tau=TAU,
                        validate_every=max(VAL_EVERY, 1),
                    ),
                    num_epochs=ABLATION_EPOCHS,
                    learning_rate=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                    run_dir=run_dir,
                    logger=logger,
                )

                # Evaluate Logical Drift
                student.to(device)
                try:
                    spearman_rho, drift = evaluate_logical_drift(
                        teacher, student, test_loader, device, num_samples=DRIFT_SAMPLES
                    )
                except Exception:
                    spearman_rho, drift = float("nan"), float("nan")

                vm = result.get("report_metrics", {})

                # Adversarial evaluation (Table IV metrics)
                try:
                    adv = evaluate_adv_metrics(
                        student,
                        teacher,
                        test_loader,
                        device,
                        num_classes=NUM_CLASSES,
                        epsilon=EPSILON,
                        num_samples=2000,
                    )
                except Exception as adv_e:
                    logger.warning(f"  Adversarial eval failed: {adv_e}")
                    adv = {"adv_f1_macro": float("nan"), "adv_fpr_macro": float("nan")}

                # Latency profiling
                lat = evaluate_latency_ms(student, device, INPUT_FEATURES)

                run_metrics[config_name] = {
                    "f1_macro":        vm.get("f1_macro", 0.0),
                    "precision_macro": vm.get("precision_macro", 0.0),
                    "recall_macro":    vm.get("recall_macro", 0.0),
                    "roc_auc_macro":   vm.get("roc_auc_macro", 0.0),
                    "fpr_macro":       vm.get("fpr_macro", 0.0),
                    "accuracy":        vm.get("accuracy", 0.0),
                    "spearman_rho":    spearman_rho,
                    "delta_L":         drift,
                    "adv_f1_macro":    adv["adv_f1_macro"],
                    "adv_fpr_macro":   adv["adv_fpr_macro"],
                    "latency_ms":      lat["latency_ms_mean"],
                }

                logger.info(
                    f"    Result: F1={run_metrics[config_name]['f1_macro']:.4f}, "
                    f"ΔL={drift:.4f}"
                )
            except Exception as e:
                logger.error(f"  {config_name} FAILED: {e}")
                traceback.print_exc()

            # Free GPU memory between configs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        all_results.append(run_metrics)

        # Save per-run
        with open(run_dir / "ablation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(run_metrics, f, indent=2, cls=NpEncoder)

    # ── Aggregate ──
    if not all_results:
        logger.error("No successful runs.")
        return

    aggregated = {}
    for config_name in ABLATION_CONFIGS:
        aggregated[config_name] = {}
        for metric in ["f1_macro", "precision_macro", "recall_macro",
                       "roc_auc_macro", "fpr_macro", "accuracy",
                       "spearman_rho", "delta_L",
                       "adv_f1_macro", "adv_fpr_macro", "latency_ms"]:
            vals = [r[config_name][metric] for r in all_results
                    if config_name in r and metric in r[config_name]]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            lo, hi = bootstrap_ci(arr) if len(arr) > 1 else (arr[0], arr[0])
            aggregated[config_name][metric] = {
                "values": arr.tolist(),
                "mean":   float(arr.mean()),
                "std":    float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "ci95":   [lo, hi],
            }

    # Save
    stats_path = RESULTS_ROOT / "ABLATION_STATS.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, cls=NpEncoder)
    logger.info(f"\nAggregated stats → {stats_path}")

    # ── Print table ──
    sep = "-" * 80
    header = f"{'Config':<16}{'F1-Macro':^16}{'ΔL (Drift)':^16}{'Adv FPR':^16}{'Latency':^14}"
    lines = [sep, header, sep]
    for config_name, mdict in aggregated.items():
        f1 = mdict.get("f1_macro", {})
        dl = mdict.get("delta_L", {})
        afpr = mdict.get("adv_fpr_macro", {})
        lat = mdict.get("latency_ms", {})
        row = f"{config_name:<16}"
        row += f"{f1.get('mean', 0)*100:.2f}±{f1.get('std', 0)*100:.2f}%".center(16)
        row += f"{dl.get('mean', 0):.4f}±{dl.get('std', 0):.4f}".center(16)
        row += f"{afpr.get('mean', 0)*100:.2f}%".center(16)
        row += f"{lat.get('mean', 0):.2f}ms".center(14)
        lines.append(row)
    lines.append(sep)
    table = "\n".join(lines)
    logger.info("\n" + table)

    table_path = RESULTS_ROOT / "ABLATION_TABLE.txt"
    table_path.write_text(table, encoding="utf-8")
    logger.info(f"Table → {table_path}")
    logger.info("\n ABLATION STUDY COMPLETE.")


if __name__ == "__main__":
    main()
