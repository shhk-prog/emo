"""
run_causal_localization_sweep.py

Purpose:
Performs a comprehensive layer-by-layer sweep across all 28 layers (Layer 0 to 27)
for Qwen2.5-1.5B-Instruct to directly test the core dissociation:
  argmax_l Decodability_l != argmax_l Causal_Influence_l
  rho(Decodability, Causal_Influence) ~= 0

For each layer l:
1. Decodability (D_l): Held-out linear probe R^2 for Valence
2. Causal Influence (C_l): Within-model substitution recovery (Peak -> Neutral)
   for (a) MLP output (last token), (b) Residual stream (last token)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from v2.src.likelihood import generate_81_candidates, compute_likelihoods_for_candidates, compute_expected_va
from v3.scripts.run_aligned_cross_model_patching import get_3way_split

def compute_2d_emd(p1, p2):
    v_marg1 = np.sum(p1, axis=1)
    v_marg2 = np.sum(p2, axis=1)
    a_marg1 = np.sum(p1, axis=0)
    a_marg2 = np.sum(p2, axis=0)
    coords = np.arange(1, 10)
    emd_v = wasserstein_distance(coords, coords, v_marg1, v_marg2)
    emd_a = wasserstein_distance(coords, coords, a_marg1, a_marg2)
    return emd_v + emd_a

def format_prompt(text):
    return f"Read the following text and report your affective state.\n\nText: {text}\n\nRespond strictly in JSON format with 'valence' and 'arousal' keys (1-9)."

def get_last_token_patch_hook(source_tensor, target_pos):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            if target_pos < h.shape[1]:
                h[0, target_pos, :] = source_tensor.to(h.dtype)
            return (h,) + output[1:]
        else:
            if target_pos < output.shape[1]:
                output[0, target_pos, :] = source_tensor.to(output.dtype)
            return output
    return hook

def extract_layer_representations(model, tokenizer, texts, layer, comp="mlp", device="cuda"):
    acts = []
    for text in texts:
        prompt = format_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        target_pos = inputs.input_ids.shape[1] - 1
        
        extracted = {}
        def hook(m, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            extracted["val"] = val[0, target_pos, :].detach().cpu().numpy()
            
        if comp == "mlp":
            handle = model.model.layers[layer].mlp.register_forward_hook(hook)
        else:
            handle = model.model.layers[layer].register_forward_hook(hook)
            
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        acts.append(extracted["val"])
    return np.array(acts)

def fit_and_eval_probe(train_acts, train_labels, test_acts, test_labels, alpha=1.0):
    probe = Ridge(alpha=alpha)
    probe.fit(train_acts, train_labels)
    preds = probe.predict(test_acts)
    
    # R^2
    ss_res = np.sum((test_labels - preds) ** 2)
    ss_tot = np.sum((test_labels - np.mean(test_labels)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
    return float(r2)

def evaluate_layer_causal_effect(model, tokenizer, valid_pairs, layer, comp, candidates, va_pairs, device="cuda"):
    recoveries = []
    shifts = []
    
    for pid, peak_row, neutral_row in valid_pairs[:15]: # 15 pairs per layer for efficiency in full sweep
        peak_prompt = format_prompt(peak_row['text'])
        neutral_prompt = format_prompt(neutral_row['text'])
        
        # Source (Peak)
        l_peak, p_peak = compute_likelihoods_for_candidates(model, tokenizer, peak_prompt, candidates)
        ev_peak, _, _, _, _ = compute_expected_va(l_peak, va_pairs)
        
        # Target (Neutral)
        l_neut, p_neut = compute_likelihoods_for_candidates(model, tokenizer, neutral_prompt, candidates)
        ev_neut, _, _, _, _ = compute_expected_va(l_neut, va_pairs)
        
        # Extract Peak activation
        extracted_act = {}
        def extract_hook(m, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            extracted_act["val"] = val[0, -1, :].detach().clone()
            
        target_mod = model.model.layers[layer].mlp if comp == "mlp" else model.model.layers[layer]
        handle = target_mod.register_forward_hook(extract_hook)
        
        peak_inputs = tokenizer(peak_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**peak_inputs)
        handle.remove()
        
        # Patch into Neutral
        neutral_inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
        neutral_last_pos = neutral_inputs.input_ids.shape[1] - 1
        
        patch_handle = target_mod.register_forward_hook(
            get_last_token_patch_hook(extracted_act["val"], neutral_last_pos)
        )
        
        l_patch, p_patch = compute_likelihoods_for_candidates(model, tokenizer, neutral_prompt, candidates)
        ev_patch, _, _, _, _ = compute_expected_va(l_patch, va_pairs)
        patch_handle.remove()
        
        # EMD Recovery
        p_source_mat = np.array(p_peak).reshape(9, 9)
        p_target_mat = np.array(p_neut).reshape(9, 9)
        p_patch_mat = np.array(p_patch).reshape(9, 9)
        
        baseline_emd = compute_2d_emd(p_target_mat, p_source_mat)
        patch_emd = compute_2d_emd(p_patch_mat, p_source_mat)
        
        rec = (1.0 - (patch_emd / baseline_emd)) if baseline_emd > 1e-6 else 0.0
        recoveries.append(rec)
        
        denom = ev_peak - ev_neut
        shift = ((ev_patch - ev_neut) / denom) if abs(denom) > 1e-4 else 0.0
        shifts.append(shift)
        
    return float(np.mean(recoveries) * 100), float(np.mean(shifts) * 100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="v3/data/aipsy_strict_expanded.csv")
    parser.add_argument("--output_path", type=str, default="v3/results/causal_localization_sweep_results.csv")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()
    
    df = pd.read_csv(args.data_path)
    train_df, _, test_df = get_3way_split(df)
    
    # Extract labels (intensity: peak=1, neutral=0 for probing)
    train_labels = (train_df['intensity'] == 'peak').astype(float).values
    test_labels = (test_df['intensity'] == 'peak').astype(float).values
    
    # Valid pairs for causal substitution
    pair_ids = test_df['pair_id'].unique()
    valid_pairs = []
    for pid in pair_ids:
        pdf = test_df[test_df['pair_id'] == pid]
        if 'peak' in pdf['intensity'].values and 'neutral' in pdf['intensity'].values:
            valid_pairs.append((
                pid,
                pdf[pdf['intensity'] == 'peak'].iloc[0],
                pdf[pdf['intensity'] == 'neutral'].iloc[0]
            ))
            
    candidates, va_pairs = generate_81_candidates()
    num_layers = len(model.model.layers)
    print(f"Starting Full Layer Sweep across {num_layers} layers...")
    
    sweep_results = []
    
    for l in range(num_layers):
        print(f"\n>>> Layer {l}/{num_layers - 1} <<<")
        # 1. Probing on MLP and Resid
        train_mlp = extract_layer_representations(model, tokenizer, train_df['text'].tolist(), l, comp="mlp", device=device)
        test_mlp = extract_layer_representations(model, tokenizer, test_df['text'].tolist(), l, comp="mlp", device=device)
        r2_mlp = fit_and_eval_probe(train_mlp, train_labels, test_mlp, test_labels)
        
        train_res = extract_layer_representations(model, tokenizer, train_df['text'].tolist(), l, comp="resid", device=device)
        test_res = extract_layer_representations(model, tokenizer, test_df['text'].tolist(), l, comp="resid", device=device)
        r2_res = fit_and_eval_probe(train_res, train_labels, test_res, test_labels)
        
        # 2. Causal Substitution on MLP and Resid
        emd_rec_mlp, shift_mlp = evaluate_layer_causal_effect(model, tokenizer, valid_pairs, l, "mlp", candidates, va_pairs, device=device)
        emd_rec_res, shift_res = evaluate_layer_causal_effect(model, tokenizer, valid_pairs, l, "resid", candidates, va_pairs, device=device)
        
        print(f"Layer {l} | MLP: Probe R^2={r2_mlp:.3f}, Rec={emd_rec_mlp:.2f}%, Shift={shift_mlp:.2f}%")
        print(f"Layer {l} | Resid: Probe R^2={r2_res:.3f}, Rec={emd_rec_res:.2f}%, Shift={shift_res:.2f}%")
        
        sweep_results.append({
            "layer": l,
            "probe_r2_mlp": r2_mlp,
            "probe_r2_resid": r2_res,
            "causal_rec_mlp": emd_rec_mlp,
            "causal_shift_mlp": shift_mlp,
            "causal_rec_resid": emd_rec_res,
            "causal_shift_resid": shift_res
        })
        
    res_df = pd.DataFrame(sweep_results)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    res_df.to_csv(args.output_path, index=False)
    print(f"\nAll-layer sweep completed. Results saved to {args.output_path}")
    
    # Correlations
    r_mlp, p_mlp = spearmanr(res_df['probe_r2_mlp'], res_df['causal_rec_mlp'])
    r_res, p_res = spearmanr(res_df['probe_r2_resid'], res_df['causal_rec_resid'])
    print(f"Spearman Correlation between Decodability and Causal Recovery:")
    print(f"  MLP:   rho = {r_mlp:.4f} (p = {p_mlp:.4f})")
    print(f"  Resid: rho = {r_res:.4f} (p = {p_res:.4f})")

if __name__ == "__main__":
    main()
