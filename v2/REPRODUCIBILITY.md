# Reproducibility Guide for "When Affective Self-Reports Do Not Trace Internal Representations"

This document provides instructions for reproducing the experiments and analyses in the paper.

## 1. Environment Setup

We recommend using `uv` for fast python environment management, but `venv` or `conda` works as well.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## 2. Dataset

The experiments use the AIPSY minimal-pair dataset for Valence forcing.
- **Raw Data**: `data/raw/` (Read-only)
- **Processed Stimuli**: `v2/data/processed/aipsy_annotated/test_strict.csv`
  
To regenerate the processed stimuli from the raw data:
```bash
python v2/scripts/prepare_aipsy_strict.py
```

## 3. Running the Experiments

To run the main evaluations and interventions across the Qwen2.5-1.5B (Base and Instruct) models:

### Preliminary Behavioral Evaluation & Probe Training
```bash
python v2/scripts/run_probing_preliminary.py
```

### Representation Transformation (Cross-Decoding)
```bash
python v2/scripts/run_strict_cross_decoding.py
```

### Path Patching & Substitution Tests
```bash
# Activation Patching Screening
python v2/scripts/run_strict_patching_screening.py

# Late-Residual Substitution Test (Table 4)
python v2/scripts/run_strict_causal_scrubbing.py

# Output-stage Substitution / Swap Analysis (Table 5)
python v2/scripts/run_unembedding_norm_swap.py
```

## 4. Prompt Templates and Hashes

All experiments (except preliminary behavioral evaluations) enforce a strictly identical prompt condition to prevent trivial tokenization or format differences from confounding the causal interventions.

The identical prompt template uses the exact string structure matching the chat template, forced up to the target token `{`.
The exact prompt hashes are saved in `v2/results/prompt_hashes.json`.

```bash
# To verify the prompt hash
python v2/scripts/generate_reproducibility_artifacts.py
```

## 5. Statistical Analyses (Bootstrap CIs)

To compute the cluster bootstrap confidence intervals (e.g., Table 4):
```bash
python v2/scripts/calculate_bootstrap_ci.py
```

## Note on Model Checkpoints
- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen2.5-1.5B-Instruct`

Ensure you have sufficient VRAM (~8GB) to run inferences, as the patching algorithms may cache intermediate activations in GPU memory.
