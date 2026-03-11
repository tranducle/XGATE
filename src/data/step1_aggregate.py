import os
import pandas as pd
import glob
import time

def aggregate_phase_1(data_dir, output_file):
    print(f"--- PHASE 1: AGGREGATION & SELECTION ---")
    print(f"Source Directory: {data_dir}")
    print(f"Target Checkpoint: {output_file}")
    
    start_time = time.time()
    
    # 1. Gather all CSV files
    all_csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    # Ignore the downsampled CSV files in the 'Selected dataset for ML and DL' folder
    csv_files = [f for f in all_csv_files if "Selected dataset for ML and DL" not in f]
    
    print(f"\nFound {len(csv_files)} raw CSV files to process.")
    
    # These are identifiers or fields dropped in the original author's script
    columns_to_drop = [
        'frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4',
        'arp.dst.proto_ipv4', 'http.file_data', 'http.request.full_uri',
        'icmp.transmit_timestamp', 'icmp.unused', 'http.tls_port',
        'tcp.payload', 'tcp.options', 'udp.payload'
    ]
    
    dataframes = []
    total_rows = 0
    
    for i, file_path in enumerate(csv_files):
        file_name = os.path.basename(file_path)
        print(f"[{i+1}/{len(csv_files)}] Loading: {file_name}...")
        
        try:
            # We use low_memory=False and dtype=str for the first pass to avoid DtypeWarnings
            # from mixed data types (like headers accidentally ingested as data in some bad rows).
            df = pd.read_csv(file_path, low_memory=False, dtype=str)
            
            # Clean up the column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()
            
            # Remove any row that is accidentally a repetition of the header row
            if 'Attack_type' in df.columns:
                 df = df[df['Attack_type'] != 'Attack_type']
            
            # Drop unnecessary identifier columns immediately to save RAM
            cols_to_drop_present = [col for col in columns_to_drop if col in df.columns]
            df = df.drop(columns=cols_to_drop_present, errors='ignore')
            
            # Drop the binary label if it exists, we only want the multi-class 'Attack_type'
            if 'Attack_label' in df.columns:
                df = df.drop(columns=['Attack_label'], errors='ignore')
                
            rows_in_file = len(df)
            total_rows += rows_in_file
            print(f"    -> Extracted {rows_in_file:,} rows. Current shape: {df.shape}")
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"    [!] Error reading {file_name}: {e}")
            return
            
    print("\nConcatenating all DataFrames. This may take a moment and use up memory...")
    try:
        master_df = pd.concat(dataframes, ignore_index=True)
        print(f"Successful Concatenation! Final Shape: {master_df.shape}")
        
        # Save to Parquet
        print(f"\nSaving Checkpoint 1 to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # FIX: Ensure absolute string types for Parquet compatibility and fix encoding issues. 
        # Some columns like `arp.opcode` have mixed types (ints and strings) or broken unicode characters.
        for col in master_df.columns:
            # We enforce conversion to python string and replace any broken unicode chars
            master_df[col] = master_df[col].astype(str).str.encode('utf-8', 'ignore').str.decode('utf-8')
            
        # Using pyarrow engine is generally more stable than fastparquet for massive dirty DataFrames
        master_df.to_parquet(output_file, engine='pyarrow', index=False)
        
        elapsed = time.time() - start_time
        print(f"\n✅ PHASE 1 COMPLETE in {elapsed:.2f} seconds!")
        print(f"Checkpoint saved at: {output_file}")
        print(f"Total Rows Verified: {len(master_df):,}")
        
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR during concatenation or saving: {e}")

if __name__ == "__main__":
    DATA_DIR = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\Edge-IIoTset dataset"
    OUTPUT_FILE = r"c:\Users\Tran Duc Le\Desktop\XGATE\XGATE\DATASET\processed\checkpoint1_merged.parquet"
    aggregate_phase_1(DATA_DIR, OUTPUT_FILE)
