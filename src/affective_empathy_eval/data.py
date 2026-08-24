import pandas as pd
import numpy as np

def scale_vad(raw_value: float) -> float:
    """Scales EmoBank 5-point rating to approximately [-1, 1]."""
    return (raw_value - 3) / 2

def load_emobank(filepath: str) -> pd.DataFrame:
    """Loads EmoBank dataset and applies V/A scaling for both reader and writer."""
    df = pd.read_csv(filepath)
    
    # Reader perspective
    df['V_reader_scaled'] = df['V'].apply(scale_vad)
    df['A_reader_scaled'] = df['A'].apply(scale_vad)
    
    # We might have writer perspective in EmoBank (often in emobank.csv as V, A, D and expected reader/writer)
    # Note: EmoBank's default corpus has V, A, D. We assume these are reader perspective for now,
    # or follow the specific EmoBank structure.
    return df

def stratify_stimuli(df: pd.DataFrame, cells_v: int = 3, cells_a: int = 3, n_per_cell: int = 50, seed: int = 42) -> pd.DataFrame:
    """Stratifies the stimuli into a VA grid and samples n_per_cell."""
    df_filtered = df.copy()
    
    # Exclude too short texts or artifacts if needed
    # For now, just bin them
    
    v_bins = np.linspace(-1, 1, cells_v + 1)
    a_bins = np.linspace(-1, 1, cells_a + 1)
    
    df_filtered['v_cell'] = pd.cut(df_filtered['V_reader_scaled'], bins=v_bins, labels=False, include_lowest=True)
    df_filtered['a_cell'] = pd.cut(df_filtered['A_reader_scaled'], bins=a_bins, labels=False, include_lowest=True)
    
    sampled = df_filtered.groupby(['v_cell', 'a_cell'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_per_cell), random_state=seed)
    )
    return sampled
