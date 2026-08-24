import pandas as pd
import numpy as np

def scale_vad(raw_value: float) -> float:
    """Scales EmoBank 5-point rating to approximately [-1, 1]."""
    return (raw_value - 3) / 2

def load_emobank(filepath: str, v_col: str = "V", a_col: str = "A") -> pd.DataFrame:
    """Loads EmoBank dataset and applies V/A scaling using explicit column names."""
    df = pd.read_csv(filepath)
    
    if v_col not in df.columns or a_col not in df.columns:
        raise ValueError(f"Columns {v_col} and/or {a_col} not found in {filepath}. Available: {df.columns}")
        
    df['V_scaled'] = df[v_col].apply(scale_vad)
    df['A_scaled'] = df[a_col].apply(scale_vad)
    
    return df

def stratify_stimuli(df: pd.DataFrame, cells_v: int = 3, cells_a: int = 3, n_per_cell: int = 50, seed: int = 42):
    """Stratifies the stimuli into a VA grid and samples n_per_cell.
    Returns: (sampled_df, report_df)
    """
    df_filtered = df.copy()
    
    v_bins = np.linspace(-1, 1, cells_v + 1)
    a_bins = np.linspace(-1, 1, cells_a + 1)
    
    df_filtered['v_cell'] = pd.cut(df_filtered['V_scaled'], bins=v_bins, labels=False, include_lowest=True)
    df_filtered['a_cell'] = pd.cut(df_filtered['A_scaled'], bins=a_bins, labels=False, include_lowest=True)
    
    report_rows = []
    sampled_blocks = []
    
    for (v_cell, a_cell), group in df_filtered.groupby(['v_cell', 'a_cell']):
        candidate_n = len(group)
        sampled_n = min(candidate_n, n_per_cell)
        shortfall_n = n_per_cell - sampled_n
        
        report_rows.append({
            'v_cell': v_cell,
            'a_cell': a_cell,
            'candidate_n': candidate_n,
            'sampled_n': sampled_n,
            'target_n': n_per_cell,
            'shortfall_n': shortfall_n
        })
        
        if sampled_n > 0:
            sampled_blocks.append(group.sample(sampled_n, random_state=seed))
            
    sampled_df = pd.concat(sampled_blocks, ignore_index=True) if sampled_blocks else pd.DataFrame()
    report_df = pd.DataFrame(report_rows)
    
    return sampled_df, report_df
