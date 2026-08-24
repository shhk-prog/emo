#!/usr/bin/env python3
import os
import yaml
import pandas as pd
import argparse
from affective_empathy_eval.data import load_emobank, stratify_stimuli

def main():
    parser = argparse.ArgumentParser(description="Prepare stimuli for the experiment.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to the config file.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    data_path = "data/raw/emobank.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run scripts/download_data.py first.")
        return
        
    df = load_emobank(data_path)
    
    n_per_cell = config["sampling"].get("samples_per_cell", 50)
    seed = config["experiment"].get("random_seed", 42)
    
    sampled = stratify_stimuli(df, n_per_cell=n_per_cell, seed=seed)
    
    # Add stimulus_id using the original id
    if 'id' in sampled.columns:
        sampled['stimulus_id'] = 'emobank_' + sampled['id'].astype(str)
    else:
        sampled['stimulus_id'] = ['emobank_' + str(i) for i in range(len(sampled))]
    
    out_path = "data/processed/stimuli.csv"
    sampled.to_csv(out_path, index=False)
    print(f"Sampled {len(sampled)} stimuli and saved to {out_path}")

if __name__ == "__main__":
    main()
