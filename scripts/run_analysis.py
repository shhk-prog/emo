#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Analyze a run and generate derived dataset.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to analyze.")
    parser.add_argument("--phase", type=str, default=None, help="Experiment phase (e.g., preliminary, main_a). If None, searches across phases.")
    args = parser.parse_args()

    if args.phase:
        run_dir = os.path.join("results/raw", args.phase, args.run_id)
    else:
        # Search for the run_id in all phases
        import glob
        matches = glob.glob(os.path.join("results/raw", "*", args.run_id))
        if not matches:
            print(f"Error: Run directory not found for {args.run_id} in any phase.")
            return
        run_dir = matches[0]
        args.phase = os.path.basename(os.path.dirname(run_dir))

    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        return
        
    responses_path = os.path.join(run_dir, "responses.jsonl")
    if not os.path.exists(responses_path):
        print(f"Error: {responses_path} not found.")
        return
        
    print(f"Loading responses from {responses_path}...")
    records = []
    with open(responses_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError:
                continue
                
    df_raw = pd.DataFrame(records)
    
    if len(df_raw) == 0:
        print("No valid JSON lines found.")
        return
        
    # Split by condition
    df_base = df_raw[df_raw['condition'] == 'baseline'].copy()
    df_post = df_raw[df_raw['condition'] == 'affective_reception'].copy()
    df_rec = df_raw[df_raw['condition'] == 'recognition'].copy()
    df_emp = df_raw[df_raw['condition'] == 'empathic_response'].copy()
    
    if len(df_post) == 0 or len(df_base) == 0:
        print("Missing baseline or affective_reception conditions in the log.")
        return

    # Scale 1-9 integer outputs to [-1, 1] space for analysis
    for df in [df_base, df_post, df_rec]:
        df['parsed_valence'] = (pd.to_numeric(df['parsed_valence'], errors='coerce') - 5) / 4.0
        df['parsed_arousal'] = (pd.to_numeric(df['parsed_arousal'], errors='coerce') - 5) / 4.0

    # Extract required columns for merging
    df_base = df_base[['baseline_id', 'parsed_valence', 'parsed_arousal']].rename(columns={
        'parsed_valence': 'baseline_V',
        'parsed_arousal': 'baseline_A'
    })
    
    # Merge Post with Baseline
    df_merged = df_post.merge(df_base, on='baseline_id', how='left')
    
    # Calculate ΔV and ΔA
    df_merged['delta_V'] = df_merged['parsed_valence'] - df_merged['baseline_V']
    df_merged['delta_A'] = df_merged['parsed_arousal'] - df_merged['baseline_A']
    
    # Calculate Reactivity Magnitude R
    df_merged['R'] = np.sqrt(df_merged['delta_V']**2 + df_merged['delta_A']**2)
    
    # Merge with stimuli to get human annotations
    stimuli_path = "data/processed/stimuli.csv"
    if os.path.exists(stimuli_path):
        df_stim = pd.read_csv(stimuli_path)
        if 'stimulus_id' in df_stim.columns:
            df_merged = df_merged.merge(df_stim, on='stimulus_id', how='left')
            
            # Calculate Directional Alignment (DA) if human values are present
            if 'V_reader_scaled' in df_merged.columns and 'A_reader_scaled' in df_merged.columns:
                def calc_da(row):
                    human_V = row['V_reader_scaled']
                    human_A = row['A_reader_scaled']
                    delta_V = row['delta_V']
                    delta_A = row['delta_A']
                    
                    human_norm = np.sqrt(human_V**2 + human_A**2)
                    delta_norm = row['R']
                    
                    # Avoid division by zero
                    if human_norm < 1e-5 or delta_norm < 1e-5:
                        return np.nan
                    
                    dot_product = (human_V * delta_V) + (human_A * delta_A)
                    return dot_product / (human_norm * delta_norm)
                
                df_merged['DA'] = df_merged.apply(calc_da, axis=1)
                
                # Flag zero-norm as exclusions
                df_merged['DA_excluded_reason'] = np.where(
                    (np.sqrt(df_merged['V_reader_scaled']**2 + df_merged['A_reader_scaled']**2) < 1e-5) | (df_merged['R'] < 0.10),
                    "zero_norm",
                    None
                )
                
                # Calculate Euclidean Distance to Human (Position Alignment)
                df_merged['Euclidean_Distance_to_Human'] = np.sqrt(
                    (df_merged['parsed_valence'] - df_merged['V_reader_scaled'])**2 + 
                    (df_merged['parsed_arousal'] - df_merged['A_reader_scaled'])**2
                )
    
    # Save derived dataset
    out_dir = os.path.join("results/derived", args.phase, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "analysis_dataset.csv")
    df_merged.to_csv(out_path, index=False)
    
    print(f"Analysis complete. Derived dataset saved to {out_path}")
    print(f"Total post samples analyzed: {len(df_merged)}")

if __name__ == "__main__":
    main()
