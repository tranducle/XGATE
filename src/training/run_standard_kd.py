import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.training.data_loader import get_dataloaders, INPUT_FEATURES, NUM_CLASSES
from src.model.SecurityBERT_Model import VanillaSecurityBERT, TinySecurityBERT

NUM_EPOCHS = 10  # Reduced explicitly to save time, 10 is enough to show convergence
LEARNING_RATE = 1e-4

def train_kd_epoch(teacher, student, dataloader, optimizer, device, epoch, alpha=0.5, temp=4.0):
    student.train()
    teacher.eval()
    total_loss = 0.0
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [TRAIN KD]")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        
        with torch.no_grad():
            logits_T = teacher(X)
            
        optimizer.zero_grad()
        logits_S = student(X)
        
        # Hard Loss
        loss_ce = criterion_ce(logits_S, y)
        
        # Soft Loss (KD)
        loss_kd = criterion_kd(
            F.log_softmax(logits_S / temp, dim=1),
            F.softmax(logits_T / temp, dim=1)
        ) * (temp * temp)
        
        loss = alpha * loss_ce + (1 - alpha) * loss_kd
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def evaluate_student(student, dataloader, device):
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = student(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    print("="*50)
    print(" TRAINING STANDARD KNOWLEDGE DISTILLATION (BASELINE)")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Execution Target: {device}")
    
    train_loader, val_loader, _ = get_dataloaders()
    
    # Load Pre-trained Teacher
    teacher = VanillaSecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES).to(device)
    teacher.load_state_dict(torch.load("results/Vanilla_SecurityBERT_Teacher_best.pth", map_location=device, weights_only=True))
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Initialize Standard KD Student
    student = TinySecurityBERT(num_classes=NUM_CLASSES, input_features=INPUT_FEATURES).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_acc = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_kd_epoch(teacher, student, train_loader, optimizer, device, epoch)
        acc = evaluate_student(student, val_loader, device)
        print(f"Epoch {epoch} | Val Accuracy: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            os.makedirs("results", exist_ok=True)
            torch.save(student.state_dict(), "results/Standard_KD_TinyStudent_best.pth")
            print(f"[*] New Best Student Saved! (Acc: {acc*100:.2f}%)")

if __name__ == "__main__":
    main()
