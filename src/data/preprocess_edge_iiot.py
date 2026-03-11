import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib

# Optional but recommended for handling extreme imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("WARNING: imbalanced-learn is not installed. Hybrid sampling will be skipped.")
    print("Install it with: pip install imbalanced-learn")

def get_sampling_strategy(y, max_samples=200000, min_samples=50000):
    """
    Dynamically generates the sampling dictionary.
    - Caps majority classes at max_samples.
    - Boosts minority classes to min_samples via SMOTE.
    """
    class_counts = y.value_counts()
    under_strategy = {}
    over_strategy = {}
    
    for cls, count in class_counts.items():
        if count > max_samples:
            under_strategy[cls] = max_samples
        if count < min_samples:
            # Cannot oversample classes with extremely low counts (e.g. < 6 neighbors for default SMOTE)
            # If a class has fewer than 6 samples, we just leave it alone or use random oversampling
            # But in Edge-IIoTset, the smallest class 'OS Fingerprinting' has ~1000 rows, so SMOTE is safe.
            over_strategy[cls] = min_samples

    return under_strategy, over_strategy

def preprocess_dataset(input_csv, output_dir, max_samples=200000, min_samples=50000):
    print(f"Loading dataset from: {input_csv}")
    print("Note: With 128GB RAM, this 1.1GB file will easily fit into memory.")
    
    # 1. Load Data
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Initial Shape: {df.shape}\n")

    # The dataset documentation mentions 'Attack_type' and 'Attack_label' 
    # as the target columns. We want to predict 'Attack_type' for multi-class.
    target_col = 'Attack_type'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    df = df.drop(columns=['Attack_label'], errors='ignore') # Drop the binary label to prevent data leakage

    # 2. Drop columns with mostly missing data (>50%) or useless identifiers
    print("Dropping irrelevant/mostly missing columns...")
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    
    # Common useless identifiers from packet flow datasets
    identifiers = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp', 'icmp.unused', 'http.tls_port']
    cols_to_drop.extend([c for c in identifiers if c in df.columns])
    
    df = df.drop(columns=set(cols_to_drop), errors='ignore')

    # 3. Clean remaining missing data
    print("Imputing missing values...")
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    categorical_cols.remove(target_col) if target_col in categorical_cols else None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Fill numeric NaNs with median (robust to outliers)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Fill categorical NaNs with 'Missing' placeholder
    for col in categorical_cols:
        df[col] = df[col].fillna("Missing_Value")

    # 4. Encoding
    print("Applying Encoders...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Convert to string to ensure LabelEncoder works properly
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Encode Target
    target_le = LabelEncoder()
    df[target_col] = target_le.fit_transform(df[target_col].astype(str))
    
    # Print Class Mapping
    print("\nClass Mapping:")
    for idx, name in enumerate(target_le.classes_):
        print(f"  {idx}: {name}")

    # 5. Hybrid Sampling (SMOTE + Undersampling)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if IMBLEARN_AVAILABLE:
        print("\nPerforming Hybrid Sampling (Undersample Majority -> SMOTE Minority)...")
        print("This may take a few minutes on your RTX 3090/CPU...")
        under_strategy, over_strategy = get_sampling_strategy(y, max_samples, min_samples)
        
        steps = []
        if under_strategy:
            print(f"  Undersampling classes to max {max_samples} rows...")
            steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)))
        
        if over_strategy:
            print(f"  Oversampling classes to min {min_samples} rows using SMOTE...")
            steps.append(('over', SMOTE(sampling_strategy=over_strategy, random_state=42)))
            
        if steps:
            pipeline = Pipeline(steps=steps)
            X_resampled, y_resampled = pipeline.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y

        print(f"Resampled Shape: {X_resampled.shape}")
    else:
        X_resampled, y_resampled = X, y

    # 6. Scaling (RobustScaler based on percentiles, prevents extreme DDoS variants from skewing gradients)
    print("\nScaling Numerical Features with RobustScaler...")
    scaler = RobustScaler()
    X_resampled[numeric_cols] = scaler.fit_transform(X_resampled[numeric_cols])

    # 7. Train/Val/Test Split
    print("Splitting dataset (70% Train, 15% Validation, 15% Test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(X_resampled, y_resampled, test_size=0.15, stratify=y_resampled, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42) # 0.1765 of 0.85 approx 0.15 of total

    # Recombine to save
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # 8. Export
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting to {output_dir}...")
    
    # We use parquet for lightning fast loading into PyTorch later
    train_file = os.path.join(output_dir, "train.parquet")
    val_file = os.path.join(output_dir, "val.parquet")
    test_file = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    test_df.to_parquet(test_file, index=False)
    
    # Save the encoders and scaler so we can decode predictions later
    artifacts = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_le,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'class_weights': compute_class_weights(y_resampled)
    }
    joblib.dump(artifacts, os.path.join(output_dir, "preprocessing_artifacts.pkl"))
    
    print("\n✅ Preprocessing Complete!")
    print(f"Train size: {train_df.shape}")
    print(f"Val size:   {val_df.shape}")
    print(f"Test size:  {test_df.shape}")
    
def compute_class_weights(y):
    """Computes weights inversely proportional to class frequencies for Focal Loss/CrossEntropy"""
    class_counts = y.value_counts().sort_index().values
    total_samples = len(y)
    num_classes = len(class_counts)
    
    # standard formula: n_samples / (n_classes * np.bincount(y))
    weights = total_samples / (num_classes * class_counts)
    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Edge-IIoTset for SecurityBERT")
    parser.add_argument("--input", type=str, default=r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\Edge-IIoTset dataset\Selected dataset for ML and DL\DNN-EdgeIIoT-dataset.csv", help="Path to raw CSV dataset")
    parser.add_argument("--output", type=str, default=r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=200000, help="Undersampling threshold for majority classes")
    parser.add_argument("--min-samples", type=int, default=50000, help="Oversampling threshold for minority classes via SMOTE")
    
    args = parser.parse_args()
    preprocess_dataset(args.input, args.output, args.max_samples, args.min_samples)
