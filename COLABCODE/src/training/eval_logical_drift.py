import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from src.training.data_loader import get_dataloaders, INPUT_FEATURES, NUM_CLASSES
from src.model.SecurityBERT_Model import VanillaSecurityBERT, TinySecurityBERT

def get_input_x_gradient(model, X, target_classes):
    """
    Computes Input X Gradient for the given model and target classes.
    """
    X.requires_grad_(True)
    if X.grad is not None:
        X.grad.zero_()
        
    logits = model(X)
    
    # Gather logits for the target classes
    gathered_logits = logits.gather(1, target_classes.view(-1, 1)).squeeze(1)
    
    # Compute gradients of the sum of gathered logits w.r.t inputs
    # We use retain_graph=True if we were doing multiple backwards, but sum() is fine
    gathered_logits.sum().backward()
    
    gradients = X.grad.detach()
    attr = gradients * X.detach()
    return attr.cpu().numpy()

def evaluate_logical_drift(teacher_model, student_model, dataloader, device, num_samples=1000):
    """
    Evaluates the Logical Drift (Delta_L) using Spearman Rank Correlation
    between the Teacher and Student Feature Attributions (InputXGradient).
    """
    teacher_model.eval()
    student_model.eval()
    
    spearman_corrs = []
    processed = 0
    
    for X, y in dataloader:
        if processed >= num_samples:
            break
            
        X = X.to(device)
        y = y.to(device)
        
        # Teacher preds
        with torch.no_grad():
            preds_T = teacher_model(X).argmax(dim=1)
            preds_S = student_model(X).argmax(dim=1)
            
        # We compute attribution only on samples where BOTH models predict correctly 
        # or at least we compute relative to Teacher's prediction
        target_classes = preds_T
        
        # We need to enable grad for attribution
        phi_T = get_input_x_gradient(teacher_model, X, target_classes)
        # Clear grad
        teacher_model.zero_grad()
        
        X.requires_grad_(False)
        phi_S = get_input_x_gradient(student_model, X, target_classes)
        student_model.zero_grad()
        
        for i in range(X.size(0)):
            flat_T = np.abs(phi_T[i]).flatten()
            flat_S = np.abs(phi_S[i]).flatten()
            
            # Use top correlations or global
            if np.std(flat_T) < 1e-9 or np.std(flat_S) < 1e-9:
                continue
                
            corr, _ = spearmanr(flat_T, flat_S)
            if not np.isnan(corr):
                spearman_corrs.append(corr)
            processed += 1
            if processed >= num_samples:
                break
    
    avg_spearman = np.mean(spearman_corrs)
    delta_L = 1.0 - avg_spearman
    return avg_spearman, delta_L

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Extracting Logic on {device}")
    
    _, val_loader, _ = get_dataloaders(batch_size=128)
    
    teacher = VanillaSecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES).to(device)
    teacher.load_state_dict(torch.load("results/Vanilla_SecurityBERT_Teacher_best.pth", map_location=device, weights_only=True))
    
    # 1. Evaluate X-GATE
    xgate = TinySecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES).to(device)
    try:
        xgate.load_state_dict(torch.load("results/X-GATE_TinyStudent_best.pth", map_location=device, weights_only=True))
        print("Evaluating X-GATE TinyStudent...")
        spearman_x, drift_x = evaluate_logical_drift(teacher, xgate, val_loader, device, num_samples=1000)
        print(f"X-GATE -> Avg Spearman: {spearman_x:.4f} | Logical Drift (Delta_L): {drift_x:.4f}")
    except Exception as e:
        print(f"X-GATE error: {e}")
        
    # 2. Evaluate Standard KD
    std_kd = TinySecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES).to(device)
    try:
        std_kd.load_state_dict(torch.load("results/Standard_KD_TinyStudent_best.pth", map_location=device, weights_only=True))
        print("Evaluating Standard KD TinyStudent...")
        spearman_s, drift_s = evaluate_logical_drift(teacher, std_kd, val_loader, device, num_samples=1000)
        print(f"Standard KD -> Avg Spearman: {spearman_s:.4f} | Logical Drift (Delta_L): {drift_s:.4f}")
    except Exception as e:
        print(f"Standard KD error: {e}")
