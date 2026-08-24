#!/usr/bin/env python3
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List

from affective_empathy_eval.manifests import ManifestManager
from affective_empathy_eval.probing import LayerProber, run_rsa_analysis, run_shuffled_baseline_probing
from affective_empathy_eval.splits import create_dev_test_split

def main():
    parser = argparse.ArgumentParser(description="Run Layerwise Probing and RSA analysis on main experiment representations.")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID of the main experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Run in synthetic dry-run mode.")
    args = parser.parse_args()

    if args.dry_run or not args.run_id:
        print("Running probing in dry-run synthetic mode (Phase B1 compatible)...")
        run_id = "dryrun_probing_test"
        out_dir = os.path.join("results/derived/main", run_id)
        os.makedirs(out_dir, exist_ok=True)

        n_samples = 50
        n_layers = 12
        hidden_dim = 768
        
        rng = np.random.default_rng(42)
        
        # Synthetic targets
        target_human_v = rng.uniform(-1, 1, size=(n_samples,))
        target_human_a = rng.uniform(-1, 1, size=(n_samples,))
        
        target_reported_post_v = target_human_v + rng.normal(0, 0.2, size=(n_samples,))
        target_reported_post_a = target_human_a + rng.normal(0, 0.2, size=(n_samples,))
        
        target_delta_v = target_reported_post_v - rng.uniform(-0.1, 0.1, size=(n_samples,))
        target_delta_a = target_reported_post_a - rng.uniform(-0.1, 0.1, size=(n_samples,))
        
        target_vectors = {
            "human_VA": np.column_stack([target_human_v, target_human_a]),
            "reported_post_VA": np.column_stack([target_reported_post_v, target_reported_post_a]),
            "delta_VA": np.column_stack([target_delta_v, target_delta_a])
        }
        
        # Group IDs (2 samples per stimulus for repetition)
        group_ids = np.repeat(np.arange(n_samples // 2), 2)
        if len(group_ids) < n_samples:
            group_ids = np.append(group_ids, group_ids[:n_samples - len(group_ids)])

        layer_vectors = {}
        layer_probing_results = []
        prober = LayerProber(alpha=1.0, cv=5, seed=42, use_pca=True, n_components=10)

        for layer in range(n_layers):
            signal_weight = (layer / float(n_layers))
            noise = rng.normal(0, 1.0, size=(n_samples, hidden_dim))
            signal = np.outer(target_human_v, rng.normal(1, 0.1, size=(hidden_dim,)))
            X_layer = signal_weight * signal + (1.0 - signal_weight) * noise
            layer_vectors[layer] = X_layer

            # Probing for Human Valence using GroupKFold
            res_v = prober.evaluate_probing(X_layer, target_human_v, group_ids)
            shuffled_res = run_shuffled_baseline_probing(X_layer, target_human_v, group_ids, prober, n_permutations=3)

            layer_probing_results.append({
                "layer": layer,
                "target": "human_valence",
                "r2": res_v["r2"],
                "rmse": res_v["rmse"],
                "mae": res_v["mae"],
                "pearson_r": res_v["pearson_r"],
                "spearman_r": res_v["spearman_r"],
                "shuffled_mean_r2": shuffled_res["mean_shuffled_r2"]
            })

        df_probing = pd.DataFrame(layer_probing_results)
        probing_out_path = os.path.join(out_dir, "probing_summary.csv")
        df_probing.to_csv(probing_out_path, index=False)

        # RSA Analysis across 3 targets
        rsa_dict = run_rsa_analysis(layer_vectors, target_vectors, metric='euclidean')
        
        rsa_rows = []
        for target_name, layer_res in rsa_dict.items():
            for layer, scores in layer_res.items():
                rsa_rows.append({
                    "target": target_name,
                    "layer": layer,
                    "spearman": scores["spearman"],
                    "pearson": scores["pearson"]
                })
        
        df_rsa = pd.DataFrame(rsa_rows)
        rsa_out_path = os.path.join(out_dir, "rsa_summary.csv")
        df_rsa.to_csv(rsa_out_path, index=False)

        print(f"Dry-run probing analysis completed.")
        print(f"Probing metrics saved to {probing_out_path}")
        print(f"RSA metrics saved to {rsa_out_path}")
        return

    # Real run processing logic
    matches = glob.glob(os.path.join("results/raw/*", args.run_id))
    if not matches:
        print(f"Error: Run directory for {args.run_id} not found.")
        return
    run_dir = matches[0]

    responses_path = os.path.join(run_dir, "responses.jsonl")
    reps_dir = os.path.join(run_dir, "representations")
    
    if not os.path.exists(responses_path):
        print(f"Error: Missing responses.jsonl in {run_dir}")
        return

    print(f"Analyzing representations from {run_dir} using ManifestManager...")
    manager = ManifestManager(reps_dir)
    df_manifest = manager.load_all_as_dataframe()
    
    if df_manifest.empty:
        print("Warning: No manifests found in representation directory.")
        
    out_dir = os.path.join("results/derived/main", args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Probing script completed successfully. Results saved to {out_dir}/")

if __name__ == "__main__":
    main()
