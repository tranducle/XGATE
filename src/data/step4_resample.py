import pandas as pd
import numpy as np
import os
import time
import joblib
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import collections

def resample_phase_4(train_file, val_file, test_file, artifacts_file, output_dir):
    print(f"--- PHASE 4: HYBRID SAMPLING (SMOTE + UNDERSAMPLING) ---")
    
    start_time = time.time()
    
    try:
        print(f"Loading Checkpoint 3 (Train Set Only): {train_file}")
        # We ONLY resample the Training set. Valid/Test sets MUST remain untouched real-world distributions!
        df_train = pd.read_parquet(train_file, engine='pyarrow')
        print(f"\nLoaded Train Shape: {df_train.shape}")
        
        # Load artifacts to decode the target classes so we know what we're balancing
        artifacts = joblib.load(artifacts_file)
        label_encoder = artifacts['label_encoder']
    except Exception as e:
        print(f"[!] Failed to load checkpoint. Error: {e}")
        return

    y_train = df_train['Attack_type']
    X_train = df_train.drop(columns=['Attack_type'])
    
    # 1. Analyze Initial Class Distribution
    print("\n1. Initial Class Distribution (Encoded):")
    class_counts = collections.Counter(y_train)
    total_rows = len(y_train)
    
    for cls_idx, count in class_counts.most_common():
        cls_name = label_encoder.inverse_transform([cls_idx])[0]
        print(f"   -> {cls_name} (Class {cls_idx}): {count:,} rows ({count/total_rows:.2%})")

    # 2. Define the Hybrid Sampling Strategy
    print("\n2. Building Hybrid Sampling Strategy...")
    # Goal: Massive majority drops to 1,000,000. Minority boosts to 100,000.
    
    under_strategy = {}
    over_strategy = {}
    
    for cls_idx, count in class_counts.items():
        if count > 1000000:
            # Undersample massive classes to 1 Million
            under_strategy[cls_idx] = 1000000
            print(f"   -> Majority class {cls_idx} target: 1,000,000 (Undersampling)")
        elif count < 100000:
            # Only oversample if it has at least 6 samples (SMOTE default k_neighbors=5 needs at least 6 samples)
            if count > 6:
                over_strategy[cls_idx] = 100000
                print(f"   -> Minority class {cls_idx} target: 100,000 (SMOTE)")
            else:
                print(f"   [!] Minority class {cls_idx} has too few samples ({count}) for SMOTE. Leaving as is.")
        else:
            print(f"   -> Class {cls_idx} inside acceptable bounds ({count}). Leaving as is.")

    # 3. Execute the Imbalanced-Learn Pipeline
    print(f"\n3. Executing Sampling Pipeline in Memory (This will heavily utilize your 128GB RAM)...")
    
    # We must use imblearn's pipeline to chain these operations correctly
    samplers = []
    
    if under_strategy:
        print("   -> Initializing RandomUnderSampler...")
        samplers.append(('undersampler', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)))
        
    if over_strategy:
        print("   -> Initializing SMOTE... (this is computationally intensive!)")
        samplers.append(('smote', SMOTE(sampling_strategy=over_strategy, random_state=42)))
        
    if not samplers:
        print("   -> No resampling needed based on distribution!")
        X_resampled, y_resampled = X_train, y_train
    else:
        pipeline = ImbPipeline(steps=samplers)
        print("   -> Running Pipeline (Undersample -> SMOTE)... please wait!")
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        
    # 4. Analyze Final Distribution
    print("\n4. Final Balanced Training Distribution:")
    final_counts = collections.Counter(y_resampled)
    final_total = len(y_resampled)
    
    for cls_idx, count in final_counts.most_common():
        cls_name = label_encoder.inverse_transform([cls_idx])[0]
        print(f"   -> {cls_name} (Class {cls_idx}): {count:,} rows")
        
    print(f"\n   -> Final Train Size: {final_total:,} rows (was {total_rows:,})")

    # 5. Save Final Export
    print("\n5. Saving Final Resampled Train Dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Re-attach target
    X_resampled['Attack_type'] = y_resampled
    
    final_train_dest = os.path.join(output_dir, 'final_balanced_train.parquet')
    X_resampled.to_parquet(final_train_dest, engine='pyarrow', index=False)
    
    # We also just visibly copy the Val/Test files so it's clear they are part of the final set
    # (or you can continue to just use checkpoint3_val.parquet directly)
    
    elapsed = time.time() - start_time
    print(f"\n✅ PHASE 4 COMPLETE in {elapsed:.2f} seconds!")
    print(f"Final Training Data Saved at: {final_train_dest}")
    print(f"Validation Data Location: {val_file}")
    print(f"Testing Data Location: {test_file}")


if __name__ == "__main__":
    TRAIN_FILE = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\checkpoint3_train.parquet"
    VAL_FILE = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\checkpoint3_val.parquet"
    TEST_FILE = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\checkpoint3_test.parquet"
    ARTIFACTS_FILE = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\preprocessing_artifacts.pkl"
    OUTPUT_DIR = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\final"
    
    resample_phase_4(TRAIN_FILE, VAL_FILE, TEST_FILE, ARTIFACTS_FILE, OUTPUT_DIR)
