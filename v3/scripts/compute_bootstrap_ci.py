import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv('v3/results/causal_localization_sweep_joint_ot.csv')
for comp in ['mlp', 'attn', 'resid']:
    cdf = df[df['component'] == comp]
    x = cdf['probe_r2'].values
    y = cdf['mean_ot_recovery'].values
    rho, p = spearmanr(x, y)
    
    # Bootstrap CI
    np.random.seed(42)
    boot_rhos = []
    n = len(x)
    for _ in range(5000):
        idx = np.random.choice(n, size=n, replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rhos.append(r)
    ci_low = np.percentile(boot_rhos, 2.5)
    ci_high = np.percentile(boot_rhos, 97.5)
    print(f"{comp.upper()}: rho = {rho:.4f} (p = {p:.4f}), 95% Bootstrap CI = [{ci_low:.4f}, {ci_high:.4f}]")
