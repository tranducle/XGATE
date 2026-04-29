"""
X-GATE Training Script: Vanilla SecurityBERT Teacher + TinySecurityBERT Student.
Usage: python -m src.training.run_xgate
"""
import torch
import torch.nn as nn
from src.training.data_loader import get_dataloaders, INPUT_FEATURES, NUM_CLASSES
from src.model.SecurityBERT_Model import VanillaSecurityBERT, TinySecurityBERT
from src.training.Trainer import ExperimentTrainer

NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

def main():
    print("="*50)
    print(" TRAINING X-GATE (Vanilla Teacher & Tiny Student)")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Execution Target: {device}")
    
    train_loader, val_loader, meta = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    
    models = {
        "Vanilla_SecurityBERT_Teacher": VanillaSecurityBERT(
            num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
        ),
        "X-GATE_TinyStudent": TinySecurityBERT(
            num_classes=NUM_CLASSES, input_features=INPUT_FEATURES
        )
    }
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f" STARTING EXPERIMENT: {name}")
        print(f"{'='*50}")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        trainer = ExperimentTrainer(
            model=model, device=device,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer,
            model_name=name, num_classes=NUM_CLASSES
        )
        
        try:
            results = trainer.train(num_epochs=NUM_EPOCHS)
            final_f1 = results.get('best_f1_macro') or results.get('metrics', {}).get('f1_macro', 0)
            print(f">>> [SUCCESS] Completed {name}. Final F1: {final_f1:.4f}")
        except Exception as e:
            print(f">>> [ERROR] Failed on {name}: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
