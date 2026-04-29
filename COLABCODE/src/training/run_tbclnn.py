"""
SOTA Lightweight Baseline Training Script (2025): TBCLNN.
Usage: python -m src.training.run_tbclnn
"""
import torch
import torch.nn as nn
from src.training.data_loader import get_dataloaders, INPUT_FEATURES, NUM_CLASSES
from src.model.baselines import TBCLNN
from src.training.Trainer import ExperimentTrainer

NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

def main():
    print("="*50)
    print(" TRAINING SOTA DL BASELINE (2025): TBCLNN")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Execution Target: {device}")
    
    train_loader, val_loader, meta = get_dataloaders()
    
    model = TBCLNN(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    trainer = ExperimentTrainer(
        model=model, device=device,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer,
        model_name="TBCLNN", num_classes=NUM_CLASSES,
        num_epochs=NUM_EPOCHS
    )
    
    try:
        results = trainer.train(num_epochs=NUM_EPOCHS)
        final_f1 = results.get('best_f1_macro') or results.get('metrics', {}).get('f1_macro', 0)
        print(f">>> [SUCCESS] Completed TBCLNN. Final F1: {final_f1:.4f}")
    except Exception as e:
        print(f">>> [ERROR] Failed on TBCLNN: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
