import pandas as pd
import numpy as np
import os
import time

def clean_phase_2(input_file, output_file):
    print(f"--- PHASE 2: CLEANING & IMPUTATION ---")
    print(f"Loading Checkpoint 1: {input_file}")
    
    start_time = time.time()
    
    try:
        # Load the massive parquet file. Since Phase 1 saved everything as strings, 
        # we load it and then cast to proper types.
        df = pd.read_parquet(input_file, engine='pyarrow')
        print(f"\nInitial Loaded Shape: {df.shape}")
        
    except Exception as e:
        print(f"[!] Failed to load checkpoint. Did Phase 1 complete successfully? Error: {e}")
        return

    # In Phase 1 we forced everything to string to survive broken csvs.
    # Now we need to convert columns back to numbers where appropriate.
    # A safe trick: convert 'nan' strings to actual np.nan
    df.replace("nan", np.nan, inplace=True)
    df.replace("None", np.nan, inplace=True)
    df.replace("", np.nan, inplace=True)
    
    # 1. Deduplication
    print("\n1. Removing exact duplicates (very common in DDoS floods)...")
    initial_rows = len(df)
    df = df.drop_duplicates()
    dupes_dropped = initial_rows - len(df)
    print(f"   -> Dropped {dupes_dropped:,} duplicate rows. New shape: {df.shape}")

    # 2. Type Inference & Column Pruning
    print("\n2. Inferring data types and pruning mostly-empty columns...")
    
    # Attempt to convert to numeric safely natively without relying on strict pandas kwargs
    for col in df.columns:
        if col != 'Attack_type':  # Keep the target purely string
            try:
                # Try direct absolute cast first
                df[col] = df[col].astype('float64')
            except ValueError:
                pass # If it fails, leave the column as strings (categorical)
            except TypeError:
                pass
            
    # Drop columns where >50% of data is missing
    missing_pct = df.isna().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    
    if cols_to_drop:
        print(f"   -> Dropping {len(cols_to_drop)} columns with >50% missing data:")
        for c in cols_to_drop:
             print(f"      - {c} ({missing_pct[c]:.1%} missing)")
        df = df.drop(columns=cols_to_drop)
    else:
        print("   -> No columns exceeded 50% missing data threshold.")

    # 3. Handle Remaining Missing Data (Imputation)
    print("\n3. Imputing remaining missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    if 'Attack_type' in categorical_cols:
        categorical_cols.remove('Attack_type')
        
    # We must cleanly drop rows where the TARGET LABEL (Attack_type) is somehow missing
    clean_target_rows = df['Attack_type'].isna().sum()
    if clean_target_rows > 0:
        print(f"   -> Dropping {clean_target_rows} rows missing 'Attack_type' label.")
        df = df.dropna(subset=['Attack_type'])

    # Numeric: Fill with Median (Robust against extreme DDoS traffic spikes)
    print(f"   -> Imputing {len(numeric_cols)} numeric columns with their medians...")
    for col in numeric_cols:
        median_val = df[col].median()
        # If the entire column was somehow empty, fill with 0
        if pd.isna(median_val): 
            median_val = 0
        df[col] = df[col].fillna(median_val)
        
        # Optimize memory by downcasting float64 to float32
        if df[col].dtype == 'float64':
             df[col] = df[col].astype('float32')

    # Categorical: Fill with literal string "Missing_Value"
    print(f"   -> Imputing {len(categorical_cols)} categorical columns with 'Missing_Value' string...")
    for col in categorical_cols:
        df[col] = df[col].fillna("Missing_Value")

    # 4. Final Validation & Export
    print("\n4. Final validation before export...")
    total_missing = df.isna().sum().sum()
    if total_missing > 0:
        print(f"   [!] WARNING: {total_missing} missing values still remain! Check data closely.")
    else:
        print("   -> Validation passed: 0 missing values remain.")
        
    print(f"\nFinal Cleaned Shape: {df.shape}")
    print("\nSaving Checkpoint 2...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, engine='pyarrow', index=False)
    
    elapsed = time.time() - start_time
    print(f"\n✅ PHASE 2 COMPLETE in {elapsed:.2f} seconds!")
    print(f"Checkpoint saved at: {output_file}")


if __name__ == "__main__":
    INPUT_FILE = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\checkpoint1_merged.parquet"
    OUTPUT_FILE = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\checkpoint2_cleaned.parquet"
    clean_phase_2(INPUT_FILE, OUTPUT_FILE)
