import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

@dataclass
class PatchPair:
    """Represents a pair of stimuli for Activation Patching experiments."""
    source_id: str
    target_id: str
    source_expected_v: float
    source_expected_a: float
    target_expected_v: float
    target_expected_a: float
    metadata: Dict[str, Any] = None


class ActivationPatcher:
    """Implements Activation Patching (Clean, Corrupted, Patched runs)."""
    def __init__(self, target_layer: int, position_type: str = "stimulus_last_token"):
        self.target_layer = target_layer
        self.position_type = position_type

    def patch_activation(
        self, target_vector: np.ndarray, source_vector: np.ndarray, patch_weight: float = 1.0
    ) -> np.ndarray:
        """Interpolates target vector towards source vector by patch_weight (0=no patch, 1=full patch)."""
        return (1.0 - patch_weight) * target_vector + patch_weight * source_vector

    @staticmethod
    def calculate_recovery(clean_metric: float, corrupted_metric: float, patched_metric: float, eps: float = 1e-5) -> float:
        """
        Calculates the recovery score for a patching intervention.
        Recovery = (Patched - Corrupted) / (Clean - Corrupted)
        Returns np.nan if the denominator (effect size between clean and corrupted) is too small.
        """
        denominator = clean_metric - corrupted_metric
        if abs(denominator) < eps:
            return np.nan
        return float((patched_metric - corrupted_metric) / denominator)


class RepresentationAblator:
    """Implements Ablation interventions, strictly focusing on Neutral replacement as main analysis."""
    def __init__(self, ablation_type: str = "neutral"):
        assert ablation_type in ["zero", "mean", "neutral"]
        self.ablation_type = ablation_type

    def ablate(self, target_vector: np.ndarray, reference_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Applies ablation. 
        For 'neutral', reference_vector should be the mean vector of neutral stimuli.
        For 'mean', reference_vector should be the mean of all stimuli.
        For 'zero', returns zero vector.
        """
        if self.ablation_type == "zero":
            return np.zeros_like(target_vector)
        elif self.ablation_type in ["mean", "neutral"]:
            if reference_vector is None:
                raise ValueError(f"reference_vector must be provided for {self.ablation_type} ablation.")
            return reference_vector.copy().astype(target_vector.dtype)
        return target_vector


class SteeringController:
    """Computes and applies affective steering vectors to model representations."""
    def __init__(self, steering_direction: np.ndarray):
        # Normalize the steering direction
        norm = np.linalg.norm(steering_direction)
        self.steering_direction = steering_direction / (norm + 1e-8)

    @classmethod
    def compute_direction_from_contrast(
        cls, high_affect_vectors: np.ndarray, low_affect_vectors: np.ndarray
    ) -> 'SteeringController':
        """Computes mean difference vector (high - low) as steering direction using training dev set."""
        mean_high = np.mean(high_affect_vectors, axis=0)
        mean_low = np.mean(low_affect_vectors, axis=0)
        direction = mean_high - mean_low
        return cls(steering_direction=direction)

    def apply_steering(self, target_vector: np.ndarray, alpha: float) -> np.ndarray:
        """
        Steers the target vector along the direction vector with strength alpha.
        alpha can be negative, zero, or positive.
        """
        return target_vector + (alpha * self.steering_direction).astype(target_vector.dtype)
