import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - older torch versions
    SDPBackend = None
    sdpa_kernel = None


@dataclass
class XGateLossConfig:
    ce_weight: float = 1.0
    kd_weight: float = 1.0
    fidelity_weight: float = 0.0
    adversarial_weight: float = 0.0
    kd_temp: float = 4.0
    epsilon: float = 0.03
    tau: float = 1.0
    validate_every: int = 1


def soft_rank(values: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable NeuralSort-style soft rank."""
    v_i = values.unsqueeze(-1)
    v_j = values.unsqueeze(-2)
    diff = (v_j - v_i) / tau
    pairwise = torch.sigmoid(diff)
    mask = 1.0 - torch.eye(values.shape[-1], device=values.device).unsqueeze(0)
    return 1.0 + (pairwise * mask).sum(dim=-1)


def fidelity_loss(phi_teacher: torch.Tensor, phi_student: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Soft-rank Spearman penalty. Zero is best, larger means worse alignment."""
    rank_teacher = soft_rank(phi_teacher.abs(), tau)
    rank_student = soft_rank(phi_student.abs(), tau)
    dim = phi_teacher.shape[-1]
    delta = (rank_teacher - rank_student) ** 2
    return ((6.0 * delta.sum(dim=-1)) / (dim * (dim ** 2 - 1))).mean()


def input_x_gradient(
    model: nn.Module,
    inputs: torch.Tensor,
    target_classes: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Input x Gradient attribution with optional higher-order graph support.
    create_graph=True keeps the student attribution path differentiable.
    """
    attribution_inputs = inputs.detach().clone().requires_grad_(True)
    # X-GATE fidelity training needs higher-order gradients through attention.
    # PyTorch's efficient SDPA kernels on CUDA do not implement the required
    # double-backward path, so we force the math kernel only for create_graph.
    if create_graph and attribution_inputs.is_cuda and sdpa_kernel is not None and SDPBackend is not None:
        attention_ctx = sdpa_kernel(backends=[SDPBackend.MATH])
    else:
        attention_ctx = nullcontext()

    with attention_ctx:
        logits = model(attribution_inputs)
    gathered = logits.gather(1, target_classes.view(-1, 1)).squeeze(1)

    gradients = torch.autograd.grad(
        outputs=gathered,
        inputs=attribution_inputs,
        grad_outputs=torch.ones_like(gathered),
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]

    attribution = gradients * attribution_inputs
    if not create_graph:
        attribution = attribution.detach()
    return attribution


def semantic_fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    semantic_mask: torch.Tensor,
    epsilon: float = 0.03,
    clamp_min: float = -5.0,
    clamp_max: float = 5.0,
) -> torch.Tensor:
    """FGSM attack constrained to the teacher-derived semantic mask."""
    attack_inputs = inputs.detach().clone().requires_grad_(True)
    logits = model(attack_inputs)
    loss = F.cross_entropy(logits, targets)

    gradients = torch.autograd.grad(
        outputs=loss,
        inputs=attack_inputs,
        only_inputs=True,
    )[0]

    delta = epsilon * semantic_mask * gradients.sign()
    return torch.clamp(inputs + delta.detach(), clamp_min, clamp_max).detach()


def evaluate_classifier(model, dataloader, device: torch.device, num_classes: int) -> Dict[str, float]:
    """Evaluate either a PyTorch model or an sklearn-like wrapper on a loader."""
    all_predictions = []
    all_probabilities = []
    all_targets = []

    is_torch_model = isinstance(model, nn.Module)
    if is_torch_model:
        model.eval()

    for batch_inputs, batch_targets in dataloader:
        all_targets.append(batch_targets.numpy())

        if is_torch_model:
            with torch.no_grad():
                logits = model(batch_inputs.to(device))
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = model.predict(batch_inputs)
            probabilities = model.predict_proba(batch_inputs)
            predictions = predictions.numpy() if torch.is_tensor(predictions) else np.asarray(predictions)
            probabilities = probabilities.numpy() if torch.is_tensor(probabilities) else np.asarray(probabilities)

        if probabilities.shape[1] < num_classes:
            padded = np.zeros((probabilities.shape[0], num_classes), dtype=np.float32)
            padded[:, :probabilities.shape[1]] = probabilities
            probabilities = padded

        all_predictions.append(predictions)
        all_probabilities.append(probabilities)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_prob = np.concatenate(all_probabilities)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    try:
        roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc_macro = 0.0

    confusion = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    false_positive = confusion.sum(axis=0) - np.diag(confusion)
    false_negative = confusion.sum(axis=1) - np.diag(confusion)
    true_positive = np.diag(confusion)
    true_negative = confusion.sum() - (false_positive + false_negative + true_positive)
    fpr_macro = float(np.mean(false_positive / (false_positive + true_negative + 1e-9)))

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_macro": float(roc_auc_macro),
        "fpr_macro": float(fpr_macro),
    }


def evaluate_adversarial_fpr(
    model: nn.Module,
    teacher: nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
    epsilon: float = 0.03,
    num_samples: int = 2000,
) -> Dict[str, float]:
    """Evaluate explanation-guided FGSM robustness on a held-out loader."""
    model.eval()
    teacher.eval()

    all_predictions = []
    all_targets = []
    processed = 0

    for batch_inputs, batch_targets in dataloader:
        if processed >= num_samples:
            break

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        teacher_attr = input_x_gradient(teacher, batch_inputs, batch_targets, create_graph=False)
        semantic_mask = teacher_attr.abs()
        semantic_mask = semantic_mask / (semantic_mask.sum(dim=-1, keepdim=True) + 1e-9)

        adversarial_inputs = semantic_fgsm_attack(model, batch_inputs, batch_targets, semantic_mask, epsilon)

        with torch.no_grad():
            predictions = model(adversarial_inputs).argmax(dim=1)

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_targets.cpu().numpy())
        processed += batch_inputs.size(0)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)

    _, _, adv_f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    confusion = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    false_positive = confusion.sum(axis=0) - np.diag(confusion)
    false_negative = confusion.sum(axis=1) - np.diag(confusion)
    true_positive = np.diag(confusion)
    true_negative = confusion.sum() - (false_positive + false_negative + true_positive)
    adv_fpr_macro = float(np.mean(false_positive / (false_positive + true_negative + 1e-9)))

    return {
        "adv_f1_macro": float(adv_f1_macro),
        "adv_fpr_macro": adv_fpr_macro,
    }


def evaluate_latency_ms(
    model: nn.Module,
    device: torch.device,
    input_dim: int,
    n_warmup: int = 50,
    n_measure: int = 200,
) -> Dict[str, float]:
    """Measure single-sample latency in milliseconds."""
    model.eval()
    dummy = torch.randn(1, input_dim, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(n_measure):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    timing_array = np.asarray(timings)
    return {
        "latency_ms_mean": float(timing_array.mean()),
        "latency_ms_p50": float(np.percentile(timing_array, 50)),
        "latency_ms_p99": float(np.percentile(timing_array, 99)),
    }


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Create a dynamic INT8 deployment copy for CPU-side evaluation."""
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu(),
        {nn.Linear},
        dtype=torch.qint8,
    )


def serialized_model_size_mb(model: nn.Module, path: Path) -> float:
    """Serialize a state dict to estimate deployment artifact size in megabytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path.stat().st_size / (1024 * 1024)


def train_xgate_variant(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    selection_loader,
    report_loader,
    device: torch.device,
    num_classes: int,
    config_name: str,
    loss_config: XGateLossConfig,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    run_dir: Path,
    logger,
) -> Dict[str, Dict[str, float]]:
    """
    Train one student variant, select checkpoints on `selection_loader`,
    then report final metrics on `report_loader`.
    """
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student.to(device)
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction="batchmean")

    best_selection_f1 = float("-inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    checkpoint_path = run_dir / f"{config_name}_best.pth"

    for epoch in range(1, num_epochs + 1):
        student.train()
        running_loss = 0.0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            with torch.no_grad():
                teacher_logits = teacher(batch_inputs)

            optimizer.zero_grad()
            student_logits = student(batch_inputs)

            loss_ce = criterion_ce(student_logits, batch_targets)
            loss_kd = criterion_kd(
                F.log_softmax(student_logits / loss_config.kd_temp, dim=1),
                F.softmax(teacher_logits / loss_config.kd_temp, dim=1),
            ) * (loss_config.kd_temp ** 2)

            total_loss = (
                loss_config.ce_weight * loss_ce
                + loss_config.kd_weight * loss_kd
            )

            teacher_attr = None
            if loss_config.fidelity_weight > 0:
                teacher_attr = input_x_gradient(teacher, batch_inputs, batch_targets, create_graph=False)
                student_attr = input_x_gradient(student, batch_inputs, batch_targets, create_graph=True)
                total_loss = total_loss + loss_config.fidelity_weight * fidelity_loss(
                    teacher_attr.detach(),
                    student_attr,
                    tau=loss_config.tau,
                )

            if loss_config.adversarial_weight > 0:
                if teacher_attr is None:
                    teacher_attr = input_x_gradient(teacher, batch_inputs, batch_targets, create_graph=False)

                semantic_mask = teacher_attr.abs()
                semantic_mask = semantic_mask / (semantic_mask.sum(dim=-1, keepdim=True) + 1e-9)
                adversarial_inputs = semantic_fgsm_attack(
                    student,
                    batch_inputs,
                    batch_targets,
                    semantic_mask,
                    epsilon=loss_config.epsilon,
                )
                adversarial_logits = student(adversarial_inputs)
                loss_adv_ce = criterion_ce(adversarial_logits, batch_targets)
                loss_adv_kd = criterion_kd(
                    F.log_softmax(adversarial_logits / loss_config.kd_temp, dim=1),
                    F.softmax(teacher_logits / loss_config.kd_temp, dim=1),
                ) * (loss_config.kd_temp ** 2)
                total_loss = total_loss + loss_config.adversarial_weight * (loss_adv_ce + loss_adv_kd)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            running_loss += float(total_loss.item())

        should_validate = epoch % max(loss_config.validate_every, 1) == 0 or epoch == num_epochs
        if should_validate:
            selection_metrics = evaluate_classifier(student, selection_loader, device, num_classes)
            logger.info(
                f"  [{config_name}] epoch {epoch}/{num_epochs} "
                f"loss={running_loss / max(len(train_loader), 1):.4f} "
                f"val_f1={selection_metrics['f1_macro']:.4f} "
                f"val_auc={selection_metrics['roc_auc_macro']:.4f}"
            )

            if selection_metrics["f1_macro"] > best_selection_f1:
                best_selection_f1 = selection_metrics["f1_macro"]
                best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
        else:
            logger.info(
                f"  [{config_name}] epoch {epoch}/{num_epochs} "
                f"loss={running_loss / max(len(train_loader), 1):.4f} (skip val)"
            )

    if best_state is not None:
        student.load_state_dict(best_state)
        torch.save(best_state, checkpoint_path)

    final_selection_metrics = evaluate_classifier(student, selection_loader, device, num_classes)
    final_report_metrics = evaluate_classifier(student, report_loader, device, num_classes)
    return {
        "best_selection_f1_macro": float(max(best_selection_f1, final_selection_metrics["f1_macro"])),
        "selection_metrics": final_selection_metrics,
        "report_metrics": final_report_metrics,
    }
