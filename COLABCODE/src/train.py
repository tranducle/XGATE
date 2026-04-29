"""
SAINT Training Pipeline (Optimized for RTX 3090 + 128GB RAM)
==============================================================
Training, validation, and evaluation for SAINT model.

Optimizations:
- AMP (Automatic Mixed Precision) for 2x speedup
- Optimized DataLoader with pin_memory and persistent workers
- Structured results directory with timestamped runs
- File + TensorBoard logging
- cuDNN benchmark for fixed input sizes
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model import SAINT, SAINTLoss, create_model


# =============================================================================
# Configuration
# =============================================================================

class TrainingConfig:
    """Training configuration optimized for SAINT v2"""
    
    def __init__(self, **kwargs):
        # Data paths - USE V2 processed data
        self.data_path = kwargs.get('data_path', r"d:\RESEARCH\2026\PAPER1\DATA\processed\cert_r42_processed_v2.pkl")
        self.results_base = kwargs.get('results_base', r"d:\RESEARCH\2026\PAPER1\results")
        
        # Model config (reduced complexity to prevent overfitting)
        self.d_model = kwargs.get('d_model', 256)  # Reduced from 320
        self.n_heads = kwargs.get('n_heads', 4)    # Reduced from 5 (must divide d_model)
        self.n_layers = kwargs.get('n_layers', 2)  # Reduced from 4
        self.d_ff = kwargs.get('d_ff', 512)        # Reduced from 1280
        self.seq_len = kwargs.get('seq_len', 30)
        self.dropout = kwargs.get('dropout', 0.3)  # Increased from 0.1
        
        # Training config
        self.batch_size = kwargs.get('batch_size', 128)  # Reduced from 512
        self.epochs = kwargs.get('epochs', 100)          # Increased for early stopping
        self.lr = kwargs.get('lr', 5e-4)                 # Increased from 1e-4
        self.weight_decay = kwargs.get('weight_decay', 1e-4)  # Increased regularization
        self.lambda_div = kwargs.get('lambda_div', 0.05)      # Reduced
        self.lambda_sparse = kwargs.get('lambda_sparse', 0.005)
        
        # Early stopping
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        
        # Focal Loss parameters
        self.use_focal_loss = kwargs.get('use_focal_loss', True)
        self.focal_alpha = kwargs.get('focal_alpha', 0.75)  # Weight for positive class
        self.focal_gamma = kwargs.get('focal_gamma', 2.0)   # Focus parameter
        
        # Hardware optimization
        self.num_workers = kwargs.get('num_workers', 8)
        self.pin_memory = kwargs.get('pin_memory', True)
        self.persistent_workers = kwargs.get('persistent_workers', True)
        self.use_amp = kwargs.get('use_amp', True)
        self.cudnn_benchmark = kwargs.get('cudnn_benchmark', True)
        
        # Gradient clipping
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
    
    def to_dict(self) -> Dict:
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


# =============================================================================
# Results Directory Manager
# =============================================================================

class ResultsManager:
    """Manages structured results directory for each training run"""
    
    def __init__(self, base_dir: str, run_name: Optional[str] = None):
        self.base_dir = Path(base_dir)
        
        # Create timestamped run directory
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_dir = self.base_dir / run_name
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.tensorboard_dir = self.logs_dir / "tensorboard"
        
        # Create directories
        for d in [self.logs_dir, self.checkpoints_dir, self.metrics_dir, self.tensorboard_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.log_file = self.logs_dir / "train.log"
        
    def save_config(self, config: TrainingConfig):
        """Save training configuration for reproducibility"""
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    def save_metrics(self, metrics: Dict, filename: str = "training_history.json"):
        """Save metrics to JSON"""
        metrics_path = self.metrics_dir / filename
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def get_checkpoint_path(self, name: str = "best_model.pt") -> Path:
        return self.checkpoints_dir / name


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Path, console_level: int = logging.INFO) -> logging.Logger:
    """Setup dual logging to file and console"""
    
    logger = logging.getLogger("SAINT")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (concise)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Dataset
# =============================================================================

class CERTDataset(Dataset):
    """PyTorch Dataset for CERT preprocessed data with optional normalization"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None):
        # Normalize if mean/std provided
        if mean is not None and std is not None:
            # Reshape for broadcasting: (1, 1, features)
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)
            sequences = (sequences - mean) / (std + 1e-8)
        
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =============================================================================
# Trainer
# =============================================================================

class SAINTTrainer:
    """Trainer class for SAINT model with hardware optimizations"""
    
    def __init__(
        self,
        model: SAINT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        results: ResultsManager,
        logger: logging.Logger
    ):
        self.config = config
        self.results = results
        self.logger = logger
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        
        # Enable cuDNN benchmark for fixed input sizes
        if config.cudnn_benchmark and self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            self.logger.info("cuDNN benchmark enabled")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )
        
        # Calculate pos_weight for class imbalance
        train_labels = train_loader.dataset.labels.numpy()
        pos_ratio = train_labels.sum() / len(train_labels)
        pos_weight = (1 - pos_ratio) / (pos_ratio + 1e-9)
        self.logger.info(f"Class imbalance - Positive ratio: {pos_ratio:.4f}, Weight: {pos_weight:.2f}")
        
        # Loss function (Focal Loss or BCE)
        if config.use_focal_loss:
            self.criterion = SAINTLoss(
                lambda_div=config.lambda_div,
                lambda_sparse=config.lambda_sparse,
                use_focal=True,
                focal_alpha=config.focal_alpha,
                focal_gamma=config.focal_gamma
            )
            self.logger.info(f"Using Focal Loss (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
        else:
            self.criterion = SAINTLoss(
                lambda_div=config.lambda_div,
                lambda_sparse=config.lambda_sparse,
                pos_weight=pos_weight
            )
            self.criterion.bce.pos_weight = self.criterion.bce.pos_weight.to(self.device)
            self.logger.info(f"Using BCE Loss with pos_weight={pos_weight:.2f}")
        
        # AMP Gradient Scaler
        self.use_amp = config.use_amp and self.device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            self.logger.info("AMP (Automatic Mixed Precision) enabled")
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(results.tensorboard_dir))
        
        # Metrics tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': [],
            'learning_rate': []
        }
        self.best_f1 = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_patience = config.early_stopping_patience
        self.epochs_without_improvement = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with AMP"""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_div_loss = 0.0
        total_sparse_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch_idx, (batch_x, batch_y) in enumerate(pbar):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Forward pass with AMP (loss computed in fp32 to avoid NaN)
            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(batch_x, return_attention=True)
                
                # Compute loss in fp32 to avoid numerical instability
                with autocast(device_type='cuda', enabled=False):
                    losses = self.criterion(
                        output['logits'].float(), 
                        batch_y,
                        self.model.all_attention_weights
                    )
                
                # Skip if NaN detected
                if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                    self.logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass with scaled gradients
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(batch_x, return_attention=True)
                losses = self.criterion(
                    output['logits'], 
                    batch_y,
                    self.model.all_attention_weights
                )
                
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            # Metrics
            total_loss += losses['total'].item()
            total_cls_loss += losses['cls'].item()
            total_div_loss += losses['div'].item()
            total_sparse_loss += losses['sparse'].item()
            
            preds = (output['probs'] > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'acc': f"{100*correct/total:.1f}%"
            })
            
            # Log batch metrics to TensorBoard (every 50 batches)
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Batch/loss', losses['total'].item(), global_step)
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / max(n_batches, 1),
            'cls_loss': total_cls_loss / max(n_batches, 1),
            'div_loss': total_div_loss / max(n_batches, 1),
            'sparse_loss': total_sparse_loss / max(n_batches, 1),
            'acc': correct / max(total, 1)
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch_x, batch_y in tqdm(self.val_loader, desc="Validating", leave=False):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(batch_x, return_attention=True)
                
                # Compute loss in fp32
                with autocast(device_type='cuda', enabled=False):
                    losses = self.criterion(
                        output['logits'].float(),
                        batch_y,
                        self.model.all_attention_weights
                    )
            else:
                output = self.model(batch_x, return_attention=True)
                losses = self.criterion(
                    output['logits'],
                    batch_y,
                    self.model.all_attention_weights
                )
            
            total_loss += losses['total'].item()
            preds = (output['probs'] > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(output['probs'].cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        tn = ((all_preds == 0) & (all_labels == 0)).sum()
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': total_loss / len(self.val_loader),
            'acc': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        }
    
    def train(self) -> Dict:
        """Full training loop"""
        
        self.logger.info("=" * 60)
        self.logger.info("SAINT Training Started")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"AMP Enabled: {self.use_amp}")
        self.logger.info(f"Batch Size: {self.config.batch_size}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Learning Rate: {self.config.lr}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        self.logger.info("=" * 60)
        
        for epoch in range(self.config.epochs):
            epoch_start = datetime.now()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log to history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['learning_rate'].append(current_lr)
            
            # Log to TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_metrics['acc'],
                'val': val_metrics['acc']
            }, epoch)
            self.writer.add_scalar('Metrics/F1', val_metrics['f1'], epoch)
            self.writer.add_scalar('Metrics/AUC', val_metrics['auc'], epoch)
            self.writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Metrics/Recall', val_metrics['recall'], epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            
            # Log to console/file
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0  # Reset counter
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_f1': self.best_f1,
                    'val_auc': val_metrics['auc'],
                    'config': self.config.to_dict()
                }
                
                torch.save(checkpoint, self.results.get_checkpoint_path('best_model.pt'))
                self.logger.info(f"  → New best model saved (F1: {self.best_f1:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    self.logger.info(f"  Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                    break
            
            # Save latest model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, self.results.get_checkpoint_path('latest_model.pt'))
            
            # Save metrics periodically
            self.results.save_metrics(self.history, 'training_history.json')
        
        # Close TensorBoard writer
        self.writer.close()
        
        self.logger.info("=" * 60)
        self.logger.info(f"Training Complete! Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
        self.logger.info("=" * 60)
        
        return self.history
    
    def evaluate_test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Final evaluation on test set"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Final Evaluation on Test Set")
        self.logger.info("=" * 60)
        
        # Load best model
        checkpoint = torch.load(self.results.get_checkpoint_path('best_model.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        
        # Evaluate
        original_val_loader = self.val_loader
        self.val_loader = test_loader
        test_metrics = self.validate()
        self.val_loader = original_val_loader
        
        # Log results
        self.logger.info(f"Test Accuracy:  {test_metrics['acc']:.4f}")
        self.logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
        self.logger.info(f"Test F1:        {test_metrics['f1']:.4f}")
        self.logger.info(f"Test AUC:       {test_metrics['auc']:.4f}")
        self.logger.info(f"Confusion Matrix: TP={test_metrics['tp']}, FP={test_metrics['fp']}, "
                        f"FN={test_metrics['fn']}, TN={test_metrics['tn']}")
        
        # Save final metrics
        self.results.save_metrics(test_metrics, 'final_test_metrics.json')
        
        return test_metrics


# =============================================================================
# Data Loading
# =============================================================================

def load_data(config: TrainingConfig, logger: logging.Logger) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load preprocessed data and create optimized DataLoaders"""
    
    logger.info(f"Loading data from {config.data_path}")
    
    with open(config.data_path, 'rb') as f:
        data = pickle.load(f)
    
    splits = data['splits']
    
    # Compute normalization statistics from training data
    train_X = splits['train'][0]
    # Reshape to (n_samples * seq_len, features) for computing stats
    train_X_flat = train_X.reshape(-1, train_X.shape[-1])
    mean = train_X_flat.mean(axis=0)
    std = train_X_flat.std(axis=0)
    
    logger.info(f"  Data normalization: mean range [{mean.min():.2f}, {mean.max():.2f}], std range [{std.min():.2f}, {std.max():.2f}]")
    
    # Create datasets with normalization
    train_dataset = CERTDataset(*splits['train'], mean=mean, std=std)
    val_dataset = CERTDataset(*splits['val'], mean=mean, std=std)
    test_dataset = CERTDataset(*splits['test'], mean=mean, std=std)
    
    # Get input dimension
    input_dim = splits['train'][0].shape[-1]
    
    logger.info(f"  Train: {len(train_dataset):,} samples")
    logger.info(f"  Val: {len(val_dataset):,} samples")
    logger.info(f"  Test: {len(test_dataset):,} samples")
    logger.info(f"  Input dim: {input_dim}")
    logger.info(f"  Sequence length: {splits['train'][0].shape[1]}")
    
    # Optimized DataLoaders
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
        'persistent_workers': config.persistent_workers if config.num_workers > 0 else False
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    logger.info(f"  DataLoader: batch_size={config.batch_size}, workers={config.num_workers}, "
               f"pin_memory={config.pin_memory}")
    
    return train_loader, val_loader, test_loader, input_dim


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training script"""
    
    # Configuration - uses optimized defaults from TrainingConfig
    config = TrainingConfig()
    
    # Setup results directory
    results = ResultsManager(config.results_base)
    
    # Setup logging
    logger = setup_logging(results.log_file)
    
    # Log system info
    logger.info("=" * 60)
    logger.info("SAINT Training Pipeline (Optimized)")
    logger.info("=" * 60)
    logger.info(f"Run directory: {results.run_dir}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Save config
    results.save_config(config)
    logger.info("Configuration saved to config.yaml")
    
    # Load data
    train_loader, val_loader, test_loader, input_dim = load_data(config, logger)
    
    # Create model
    model = create_model(input_dim, config={
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'n_layers': config.n_layers,
        'd_ff': config.d_ff,
        'seq_len': config.seq_len,
        'dropout': config.dropout
    })
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create trainer
    trainer = SAINTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        results=results,
        logger=logger
    )
    
    # Train
    history = trainer.train()
    
    # Final evaluation on test set
    test_metrics = trainer.evaluate_test(test_loader)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Results saved to: {results.run_dir}")
    logger.info("=" * 60)
    
    return history, test_metrics


if __name__ == "__main__":
    main()
