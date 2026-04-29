import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, LabelEncoder

def scale_and_encode_phase_3(input_file, output_dir):
    print(f"--- PHASE 3: ZERO-LEAKAGE SPLIT, SCALING & ENCODING ---")
    print(f"Loading Checkpoint 2: {input_file}")
    
    start_time = time.time()
    
    try:
        df = pd.read_parquet(input_file, engine='pyarrow')
        print(f"\nLoaded Cleaned Shape: {df.shape}")
    except Exception as e:
        print(f"[!] Failed to load checkpoint. Error: {e}")
        return

    # --- ARCHITECTURAL FIX (Data Leakage Prevention) ---
    # According to the DataPreprocessingEngineer standard, we MUST split the dataset 
    # BEFORE fitting any scalers or encoders. Doing it before the split causes Data Leakage 
    # where the model learns the distribution of the Test data.
    
    print("\n1. Performing Strict Train (70%) / Val (15%) / Test (15%) Split FIRST...")
    
    # We maintain strict stratification so all rare attacks are properly distributed
    y = df['Attack_type']
    X = df.drop(columns=['Attack_type'])
    
    # First split: 70% Train, 30% Temp (Val/Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # Second split: Split the 30% Temp evenly into 15% Val and 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"   -> Train: {X_train.shape[0]:,} rows")
    print(f"   -> Val:   {X_val.shape[0]:,} rows")
    print(f"   -> Test:  {X_test.shape[0]:,} rows")
    
    # 2. Identify Column Types
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    
    print(f"\n2. Initializing Preprocessors...")
    print(f"   -> Numerical columns to scale: {len(numeric_cols)}")
    print(f"   -> Categorical columns to encode: {len(categorical_cols)}")
    
    # RobustScaler immunizes against massive DDoS integer spike outliers
    scaler = RobustScaler()
    
    # OrdinalEncoder transforms categorical strings into integers (perfect for PyTorch nn.Embedding)
    # We use handle_unknown to map unseen test-set categories to -1 (the padding index)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # LabelEncoder strictly for the Target (y)
    label_encoder = LabelEncoder()
    
    # 3. Fit on TRAIN ONLY, Transform ALL
    print("\n3. Fitting preprocessors on TRAIN SET ONLY (Zero Leakage Protocol)...")
    
    if numeric_cols:
        print("   -> Fitting and transforming numeric columns...")
        # Fit on train ONLY
        scaler.fit(X_train[numeric_cols])
        # Transform all
        X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        # Optimize memory to float32 for PyTorch compatibility
        for col in numeric_cols:
             X_train[col] = X_train[col].astype('float32')
             X_val[col] = X_val[col].astype('float32')
             X_test[col] = X_test[col].astype('float32')
             
    if categorical_cols:
        print("   -> Fitting and transforming categorical columns...")
        # Fit on train ONLY
        encoder.fit(X_train[categorical_cols])
        # Transform all
        X_train[categorical_cols] = encoder.transform(X_train[categorical_cols])
        X_val[categorical_cols] = encoder.transform(X_val[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])
        
        # Optimize memory to int32 
        for col in categorical_cols:
            X_train[col] = X_train[col].astype('int32')
            X_val[col] = X_val[col].astype('int32')
            X_test[col] = X_test[col].astype('int32')

    print("   -> Fitting and transforming Attack_type labels...")
    label_encoder.fit(y_train)
    
    # If Val/Test have attacks not seen in train (very rare with stratification), we must safely map them
    # But stratify ensures we don't have this issue.
    def safe_transform_labels(y_data, le):
        classes = list(le.classes_)
        # map unknown to an 'Unknown' category class index if needed, otherwise drop/ignore
        known_mask = y_data.isin(classes)
        mapped = y_data.copy()
        mapped[~known_mask] = classes[0] # Fallback to normal if unseen
        return le.transform(mapped)
        
    y_train = label_encoder.transform(y_train)
    y_val = safe_transform_labels(y_val, label_encoder)
    y_test = safe_transform_labels(y_test, label_encoder)

    # Re-attach the encoded labels immediately so we have complete dataframes
    X_train['Attack_type'] = y_train
    X_val['Attack_type'] = y_val
    X_test['Attack_type'] = y_test

    # 4. Save the Preprocessed Checkpoints & Artifacts
    print("\n4. Saving final scaled & encoded datasets...")
    os.makedirs(output_dir, exist_ok=True)
    
    train_dest = os.path.join(output_dir, 'checkpoint3_train.parquet')
    val_dest = os.path.join(output_dir, 'checkpoint3_val.parquet')
    test_dest = os.path.join(output_dir, 'checkpoint3_test.parquet')
    
    X_train.to_parquet(train_dest, engine='pyarrow', index=False)
    X_val.to_parquet(val_dest, engine='pyarrow', index=False)
    X_test.to_parquet(test_dest, engine='pyarrow', index=False)
    
    print("   -> Saving Preprocessor Artifacts (Scaler, Encoder, LabelEncoder)...")
    artifact_dest = os.path.join(output_dir, 'preprocessing_artifacts.pkl')
    joblib.dump({
        'scaler': scaler,
        'encoder': encoder,
        'label_encoder': label_encoder,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }, artifact_dest)
    
    elapsed = time.time() - start_time
    print(f"\n✅ PHASE 3 COMPLETE in {elapsed:.2f} seconds!")
    print(f"Artifacts saved to: {artifact_dest}")

if __name__ == "__main__":
    INPUT_FILE = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\checkpoint2_cleaned.parquet"
    OUTPUT_DIR = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed"
    scale_and_encode_phase_3(INPUT_FILE, OUTPUT_DIR)
