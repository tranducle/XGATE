"""
SOTA Transformer Hybrid Baseline Training Script (2025): MBConv_ViT.
Usage: python -m src.training.run_mbconv_vit
"""
import torch
import torch.nn as nn
from src.training.data_loader import get_dataloaders, INPUT_FEATURES, NUM_CLASSES
from src.model.baselines import MBConv_ViT_1D
from src.training.Trainer import ExperimentTrainer

NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

def main():
    print("="*50)
    print(" TRAINING SOTA TRANSFORMER BASELINE (2025): MBConv_ViT")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Execution Target: {device}")
    
    train_loader, val_loader, meta = get_dataloaders()
    
    model = MBConv_ViT_1D(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    trainer = ExperimentTrainer(
        model=model, device=device,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer,
        model_name="MBConv_ViT", num_classes=NUM_CLASSES,
        num_epochs=NUM_EPOCHS
    )
    
    try:
        results = trainer.train(num_epochs=NUM_EPOCHS)
        final_f1 = results.get('best_f1_macro') or results.get('metrics', {}).get('f1_macro', 0)
        print(f">>> [SUCCESS] Completed MBConv_ViT. Final F1: {final_f1:.4f}")
    except Exception as e:
        print(f">>> [ERROR] Failed on MBConv_ViT: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
