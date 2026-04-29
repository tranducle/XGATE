"""
Centralized Data Loader for X-GATE Training Pipeline.
Loads preprocessed Parquet datasets and converts them to PyTorch DataLoaders.
This module is the SINGLE SOURCE OF TRUTH for data paths and loading logic.
"""
import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# CANONICAL DATA PATHS (from 4-Step Preprocessing Pipeline)
# ============================================================
PROJECT_PUBLIC_DIR = Path(__file__).resolve().parents[2]
RELATED_DATA_DIR = PROJECT_PUBLIC_DIR.parent


def _candidate_dataset_dirs():
    env_dir = os.environ.get("XGATE_DATASET_DIR")
    if env_dir:
        yield Path(env_dir)

    colab_root = os.environ.get("XGATE_COLAB_ROOT")
    if colab_root:
        yield Path(colab_root) / "DATASET" / "processed"

    yield PROJECT_PUBLIC_DIR / "DATASET" / "processed"
    yield RELATED_DATA_DIR / "DATASET" / "processed"
    yield PROJECT_PUBLIC_DIR.parent.parent / "DATASET" / "processed"


def resolve_dataset_dir() -> Path:
    for candidate in _candidate_dataset_dirs():
        candidate = candidate.resolve()
        train_file = candidate / "final" / "final_balanced_train.parquet"
        val_file = candidate / "checkpoint3_val.parquet"
        test_file = candidate / "checkpoint3_test.parquet"
        if train_file.exists() and val_file.exists() and test_file.exists():
            return candidate

    inspected = "\n".join(f"  - {candidate}" for candidate in _candidate_dataset_dirs())
    raise FileNotFoundError(
        "Unable to locate the processed X-GATE dataset.\n"
        "Searched these locations:\n"
        f"{inspected}\n"
        "Set XGATE_DATASET_DIR to the processed dataset directory if needed.\n"
        "For the provided Colab bundle, the recommended layout is\n"
        "  /content/drive/MyDrive/XGATE_COLAB/DATASET/processed"
    )


DATASET_DIR = resolve_dataset_dir()
TRAIN_FILE = DATASET_DIR / "final" / "final_balanced_train.parquet"
VAL_FILE = DATASET_DIR / "checkpoint3_val.parquet"
TEST_FILE = DATASET_DIR / "checkpoint3_test.parquet"

# ============================================================
# CONSTANTS
# ============================================================
TARGET_COLUMN = "Attack_type"
NUM_CLASSES = 15
INPUT_FEATURES = 49  # 50 columns - 1 target = 49 features
DEFAULT_BATCH_SIZE = 512


def load_parquet_to_tensors(parquet_path):
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


def _build_loader(features_np, labels_tensor, batch_size: int, shuffle: bool) -> DataLoader:
    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    dataset = TensorDataset(features_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def get_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_val_as_test: bool = False,
    include_test_loader: bool = False,
):
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

    X_val, y_val = load_parquet_to_tensors(VAL_FILE)
    print(f"    [Val]   X: {X_val.shape}, y: {y_val.shape}")

    X_test, y_test = load_parquet_to_tensors(TEST_FILE)
    print(f"    [Test]  X: {X_test.shape}, y: {y_test.shape}")

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
    X_val_np = X_val.numpy()
    X_test_np = X_test.numpy()

    scaler.fit(X_train_np)

    def _transform(array):
        return np.clip(scaler.transform(array).astype(np.float32), -5.0, 5.0)

    X_train_np = _transform(X_train_np)
    X_val_np = _transform(X_val_np)
    X_test_np = _transform(X_test_np)

    print(
        f"    Post-normalization X_train: min={X_train_np.min():.4f}, max={X_train_np.max():.4f}, "
        f"mean={X_train_np.mean():.4f}, std={X_train_np.std():.4f}"
    )

    train_loader = _build_loader(X_train_np, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(X_val_np, y_val, batch_size=batch_size, shuffle=False)
    test_loader = _build_loader(X_test_np, y_test, batch_size=batch_size, shuffle=False)

    print(f">>> DataLoaders created (Batch Size: {batch_size})\n")

    metadata = {
        "dataset_dir": str(DATASET_DIR),
        "input_features": INPUT_FEATURES,
        "num_classes": NUM_CLASSES,
        "train_samples": len(X_train_np),
        "val_samples": len(X_val_np),
        "test_samples": len(X_test_np),
        "batch_size": batch_size,
    }

    if include_test_loader:
        return train_loader, val_loader, test_loader, metadata

    eval_loader = test_loader if use_val_as_test else val_loader
    metadata["eval_samples"] = metadata["test_samples"] if use_val_as_test else metadata["val_samples"]
    metadata["eval_split"] = "test" if use_val_as_test else "val"
    return train_loader, eval_loader, metadata


if __name__ == "__main__":
    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=64, include_test_loader=True)
    print(f"Metadata: {meta}")

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        for X_batch, y_batch in loader:
            print(f"{name}: X={X_batch.shape}, y={y_batch.shape}")
            print(f"  X range: [{X_batch.min():.4f}, {X_batch.max():.4f}]")
            print(f"  y unique: {y_batch.unique().tolist()}")
            break
