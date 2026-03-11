"""
Centralized Data Loader for X-GATE Training Pipeline.
Loads preprocessed Parquet datasets and converts them to PyTorch DataLoaders.
This module is the SINGLE SOURCE OF TRUTH for data paths and loading logic.
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# CANONICAL DATA PATHS (from 4-Step Preprocessing Pipeline)
# ============================================================
DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "DATASET", "processed"
)

TRAIN_FILE = os.path.join(DATASET_DIR, "final", "final_balanced_train.parquet")
VAL_FILE = os.path.join(DATASET_DIR, "checkpoint3_val.parquet")
TEST_FILE = os.path.join(DATASET_DIR, "checkpoint3_test.parquet")

# ============================================================
# CONSTANTS
# ============================================================
TARGET_COLUMN = "Attack_type"
NUM_CLASSES = 15
INPUT_FEATURES = 49  # 50 columns - 1 target = 49 features
DEFAULT_BATCH_SIZE = 512


def load_parquet_to_tensors(parquet_path: str):
    """
    Loads a Parquet file and splits it into feature tensor X and label tensor y.
    Handles dtype conversion to float32 for features and int64 for labels.
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    
    y = df[TARGET_COLUMN].values.astype(np.int64)
    X = df.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
    
    # Safety checks
    assert X.shape[1] == INPUT_FEATURES, (
        f"Expected {INPUT_FEATURES} features, got {X.shape[1]}. "
        f"Columns: {list(df.drop(columns=[TARGET_COLUMN]).columns)}"
    )
    assert not np.isnan(X).any(), "NaN detected in features!"
    assert not np.isinf(X).any(), "Inf detected in features!"
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return X_tensor, y_tensor


def get_dataloaders(batch_size: int = DEFAULT_BATCH_SIZE, use_val_as_test: bool = False):
    """
    Returns train_loader and val_loader (or test_loader) ready for training.
    Applies StandardScaler normalization (fit on train ONLY) to ensure all features
    are centered around 0 with unit variance.
    
    Args:
        batch_size: Batch size for DataLoaders.
        use_val_as_test: If True, returns test_loader instead of val_loader.
    
    Returns:
        train_loader, eval_loader, metadata dict
    """
    from sklearn.preprocessing import StandardScaler
    
    print(">>> Loading Datasets from Parquet into RAM...")
    
    # Verify files exist
    for label, path in [("Train", TRAIN_FILE), ("Val", VAL_FILE), ("Test", TEST_FILE)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[!] {label} data not found at: {path}\n"
                f"    Run the preprocessing pipeline (step1→step4) first."
            )
    
    X_train, y_train = load_parquet_to_tensors(TRAIN_FILE)
    print(f"    [Train] X: {X_train.shape}, y: {y_train.shape}")
    
    eval_file = TEST_FILE if use_val_as_test else VAL_FILE
    eval_label = "Test" if use_val_as_test else "Val"
    X_eval, y_eval = load_parquet_to_tensors(eval_file)
    print(f"    [{eval_label}]   X: {X_eval.shape}, y: {y_eval.shape}")
    
    # ============================================================
    # CRITICAL: Runtime StandardScaler Normalization
    # The RobustScaler from preprocessing is insufficient for extreme
    # network traffic outliers (tcp.seq max=715M, tcp.ack max=89M).
    # We apply StandardScaler (zero-mean, unit-variance) fit ONLY on
    # training data to prevent data leakage.
    # ============================================================
    print(">>> Applying StandardScaler normalization (fit on train ONLY)...")
    
    scaler = StandardScaler()
    X_train_np = X_train.numpy()
    X_eval_np = X_eval.numpy()
    
    scaler.fit(X_train_np)
    X_train_np = scaler.transform(X_train_np).astype(np.float32)
    X_eval_np = scaler.transform(X_eval_np).astype(np.float32)
    
    # Clip extreme values after scaling (beyond 5 sigma is almost certainly noise)
    X_train_np = np.clip(X_train_np, -5.0, 5.0)
    X_eval_np = np.clip(X_eval_np, -5.0, 5.0)
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_eval = torch.tensor(X_eval_np, dtype=torch.float32)
    
    print(f"    Post-normalization X_train: min={X_train.min():.4f}, max={X_train.max():.4f}, "
          f"mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    
    # Create DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    eval_ds = TensorDataset(X_eval, y_eval)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    print(f">>> DataLoaders created (Batch Size: {batch_size})\n")
    
    metadata = {
        "input_features": INPUT_FEATURES,
        "num_classes": NUM_CLASSES,
        "train_samples": len(X_train),
        "eval_samples": len(X_eval),
        "batch_size": batch_size
    }
    
    return train_loader, eval_loader, metadata


if __name__ == "__main__":
    # Quick verification
    train_loader, val_loader, meta = get_dataloaders(batch_size=64)
    print(f"Metadata: {meta}")
    
    for X_batch, y_batch in train_loader:
        print(f"First batch: X={X_batch.shape}, y={y_batch.shape}")
        print(f"  X range: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
        print(f"  y unique: {y_batch.unique().tolist()}")
        break
