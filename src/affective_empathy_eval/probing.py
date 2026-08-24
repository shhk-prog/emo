import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from .metrics import compute_rsa_similarity
from .splits import get_grouped_kfold_splits

class LayerProber:
    """Evaluates how well a specific hidden representation layer predicts target affective values."""
    def __init__(self, alpha: float = 1.0, cv: int = 5, seed: int = 42, 
                 use_pca: bool = True, n_components: int = 50):
        self.alpha = alpha
        self.cv = cv
        self.seed = seed
        self.use_pca = use_pca
        self.n_components = n_components

    def _build_pipeline(self) -> Pipeline:
        steps = [('scaler', StandardScaler())]
        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components, random_state=self.seed)))
        steps.append(('ridge', Ridge(alpha=self.alpha)))
        return Pipeline(steps)

    def evaluate_probing(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray) -> Dict[str, float]:
        """
        Runs GroupKFold cross-validation pipeline to predict target y (1D array) from features X (2D array).
        Normalizes and (optionally) runs PCA inside the fold to prevent data leakage.
        Returns dictionary of metrics: r2, mse, rmse, mae, pearson_r, spearman_r.
        """
        if len(X) < self.cv or len(y) < self.cv:
            return {"r2": 0.0, "mse": 0.0, "rmse": 0.0, "mae": 0.0, "pearson_r": 0.0, "spearman_r": 0.0}

        # Create a dummy dataframe to leverage our splits.py utility
        df = pd.DataFrame({'stimulus_id': group_ids})
        
        y_preds = np.zeros_like(y, dtype=float)

        for train_idx, val_idx in get_grouped_kfold_splits(df, n_splits=self.cv, group_col='stimulus_id'):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self._build_pipeline()
            model.fit(X_train, y_train)
            y_preds[val_idx] = model.predict(X_val)

        r2 = float(r2_score(y, y_preds))
        mse = float(mean_squared_error(y, y_preds))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y, y_preds))
        
        pr, _ = pearsonr(y, y_preds)
        sr, _ = spearmanr(y, y_preds)

        return {
            "r2": r2 if not np.isnan(r2) else 0.0,
            "mse": mse if not np.isnan(mse) else 0.0,
            "rmse": rmse if not np.isnan(rmse) else 0.0,
            "mae": mae if not np.isnan(mae) else 0.0,
            "pearson_r": float(pr) if not np.isnan(pr) else 0.0,
            "spearman_r": float(sr) if not np.isnan(sr) else 0.0
        }

def compute_distance_matrix(vectors: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """Computes pairwise distance matrix for a set of vectors. Supports 'euclidean' and 'correlation'."""
    dist_array = pdist(vectors, metric=metric)
    return squareform(dist_array)

def run_rsa_analysis(
    layer_vectors: Dict[int, np.ndarray],
    target_vectors: Dict[str, np.ndarray],
    metric: str = 'euclidean'
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Computes Representational Similarity Analysis (RSA) across layers for multiple targets.
    
    Args:
        layer_vectors: Dict of layer_idx -> activation vectors.
        target_vectors: Dict of target_name -> target representation vectors.
                        e.g., {'human_VA': ..., 'reported_post_VA': ..., 'delta_VA': ...}
        metric: Distance metric to use ('euclidean' or 'correlation').
                        
    Returns:
        Dict: target_name -> {layer_idx -> {spearman: float, pearson: float}}
    """
    results: Dict[str, Dict[int, Dict[str, float]]] = {target_name: {} for target_name in target_vectors.keys()}
    
    # Precompute target distance matrices
    target_dist_mats = {
        name: compute_distance_matrix(vecs, metric=metric)
        for name, vecs in target_vectors.items()
    }
    
    for layer_idx, X_layer in layer_vectors.items():
        layer_dist_mat = compute_distance_matrix(X_layer, metric=metric)
        
        for target_name, target_dist_mat in target_dist_mats.items():
            rsa_scores = compute_rsa_similarity(layer_dist_mat, target_dist_mat)
            results[target_name][layer_idx] = rsa_scores
            
    return results

def run_shuffled_baseline_probing(
    X: np.ndarray, y: np.ndarray, group_ids: np.ndarray, prober: LayerProber, n_permutations: int = 10
) -> Dict[str, float]:
    """
    Runs permutation test by shuffling labels across groups to compute baseline random chance expectation.
    Maintains group structure during shuffling if possible, or shuffles y securely.
    """
    r2_scores = []
    rng = np.random.default_rng(prober.seed)
    
    for _ in range(n_permutations):
        # Create a shuffled copy of y
        y_shuffled = rng.permutation(y)
        res = prober.evaluate_probing(X, y_shuffled, group_ids)
        r2_scores.append(res["r2"])
        
    return {
        "mean_shuffled_r2": float(np.mean(r2_scores)),
        "std_shuffled_r2": float(np.std(r2_scores))
    }
