import pandas as pd
import numpy as np

df = pd.read_csv("v3/results/focused_causal_sweep_39pairs_pair_level.csv")
np.random.seed(42)
layers = [10, 14, 15, 18, 20, 24]
comps = ['mlp', 'attn', 'resid']

for l in layers:
    for c in comps:
        col = f"gen_rec_L{l}_{c}"
        if col in df.columns:
            v = df[col].dropna().values
            n = len(v)
            boots = [np.mean(np.random.choice(v, size=n, replace=True)) for _ in range(2000)]
            ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
            print(f"L{l}_{c:5s}: Mean={np.mean(v):+6.2f}%, Med={np.median(v):+6.2f}%, 95% CI=[{ci_low:+6.1f}%, {ci_high:+6.1f}%]")
