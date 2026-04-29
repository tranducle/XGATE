"""
Traditional ML Baselines Training Script: LightGBM & RandomForest.
Usage: python -m src.training.run_ml_baselines
"""
import torch.nn as nn
from src.training.data_loader import get_dataloaders, NUM_CLASSES
from src.model.baselines import MLBaselineWrapper
from src.training.Trainer import ExperimentTrainer
import torch

def main():
    print("="*50)
    print(" TRAINING TRADITIONAL ML BASELINES (LightGBM & RF)")
    print("="*50)
    
    device = torch.device("cpu")  # ML models run on CPU
    
    train_loader, val_loader, meta = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    
    models = {
        "LightGBM": MLBaselineWrapper(model_type='lightgbm'),
        "RandomForest": MLBaselineWrapper(model_type='rf')
    }
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f" STARTING EXPERIMENT: {name}")
        print(f"{'='*50}")
        
        trainer = ExperimentTrainer(
            model=model, device=device,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=None,
            model_name=name, num_classes=NUM_CLASSES,
            num_epochs=1
        )
        
        try:
            results = trainer.train(num_epochs=1)
            final_f1 = results.get('best_f1_macro') or results.get('metrics', {}).get('f1_macro', 0)
            print(f">>> [SUCCESS] Completed {name}. Final F1: {final_f1:.4f}")
        except Exception as e:
            print(f">>> [ERROR] Failed on {name}: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
