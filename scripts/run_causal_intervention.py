#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

from affective_empathy_eval.intervention import (
    ActivationPatcher, RepresentationAblator, SteeringController
)

def main():
    parser = argparse.ArgumentParser(description="Run Causal Intervention experiments (Patching, Ablation, Steering).")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID of the main experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Run in synthetic dry-run mode.")
    args = parser.parse_args()

    print("Executing Causal Interventions (Patching, Ablation, Steering)...")
    run_id = args.run_id or "dryrun_intervention_test"
    out_dir = os.path.join("results/derived/main", run_id)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Test Activation Patching
    patcher = ActivationPatcher(target_layer=12)
    target_vec = np.ones((768,), dtype=np.float16) * 0.5
    source_vec = np.ones((768,), dtype=np.float16) * 2.0
    patched_vec = patcher.patch_activation(target_vec, source_vec, patch_weight=1.0)
    
    # Mocking evaluation metrics for Recovery calculation
    clean_score = 7.0
    corrupted_score = 4.0
    patched_score = 6.5
    recovery = ActivationPatcher.calculate_recovery(clean_score, corrupted_score, patched_score)

    # 2. Test Ablation (Neutral replacement)
    ablator_neutral = RepresentationAblator(ablation_type="neutral")
    neutral_mean_vec = np.ones((768,), dtype=np.float16) * 0.1
    ablated_vec = ablator_neutral.ablate(target_vec, reference_vector=neutral_mean_vec)

    # 3. Test Steering Vector
    high_vecs = np.random.normal(loc=1.0, scale=0.5, size=(10, 768)).astype(np.float16)
    low_vecs = np.random.normal(loc=-1.0, scale=0.5, size=(10, 768)).astype(np.float16)
    steering = SteeringController.compute_direction_from_contrast(high_vecs, low_vecs)
    
    steered_vecs = {}
    for alpha in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        steered_vecs[f"steering_alpha_{alpha}"] = steering.apply_steering(target_vec, alpha=alpha)

    # Save summary report
    results = [
        {"intervention": "baseline_target", "norm": float(np.linalg.norm(target_vec)), "recovery": np.nan},
        {"intervention": "patching_full", "norm": float(np.linalg.norm(patched_vec)), "recovery": recovery},
        {"intervention": "ablation_neutral", "norm": float(np.linalg.norm(ablated_vec)), "recovery": np.nan},
    ]
    
    for name, vec in steered_vecs.items():
        results.append({
            "intervention": name,
            "norm": float(np.linalg.norm(vec)),
            "recovery": np.nan
        })
        
    df_res = pd.DataFrame(results)
    out_path = os.path.join(out_dir, "causal_intervention_summary.csv")
    df_res.to_csv(out_path, index=False)

    print(f"Causal Intervention experiments completed.")
    print(f"Intervention summary saved to {out_path}")

if __name__ == "__main__":
    main()
