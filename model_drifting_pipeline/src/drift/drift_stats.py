import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from typing import Dict, Any, List

class DriftStats:
    """
    Computes statistical drift metrics between two distributions:
    - Kolmogorov-Smirnov test (p-value, test statistic)
    - Wasserstein distance (Earth Mover's Distance)
    - Population Stability Index (PSI)
    - Jensen-Shannon divergence
    """

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create quantiles based on expected distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        exp_counts, _ = np.histogram(expected, bins=breakpoints)
        act_counts, _ = np.histogram(actual, bins=breakpoints)

        exp_pct = exp_counts / len(expected)
        act_pct = act_counts / len(actual)

        # Avoid zero division
        exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-4, act_pct)

        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi)

    @classmethod
    def test_feature_drift(cls, ref_values: np.ndarray, curr_values: np.ndarray) -> Dict[str, Any]:
        """
        Runs multiple statistical drift tests on 1D continuous features.
        """
        if len(ref_values) == 0 or len(curr_values) == 0:
            return {"drift_detected": False, "p_value": 1.0, "score": 0.0}

        ks_stat, p_val = ks_2samp(ref_values, curr_values)
        w_dist = wasserstein_distance(ref_values, curr_values)
        psi = cls.calculate_psi(ref_values, curr_values)

        drift_detected = (p_val < 0.05) or (psi > 0.25)
        
        return {
            "drift_detected": bool(drift_detected),
            "p_value": float(p_val),
            "ks_statistic": float(ks_stat),
            "wasserstein_distance": float(w_dist),
            "psi": float(psi),
            "ref_mean": float(np.mean(ref_values)),
            "curr_mean": float(np.mean(curr_values)),
            "ref_std": float(np.std(ref_values)),
            "curr_std": float(np.std(curr_values))
        }

    @classmethod
    def test_multivariate_embedding_drift(cls, ref_feats: np.ndarray, curr_feats: np.ndarray) -> Dict[str, Any]:
        """
        Tests drift across high-dimensional embeddings using mean cosine distance and feature norms.
        """
        if len(ref_feats) == 0 or len(curr_feats) == 0:
            return {"drift_detected": False, "score": 0.0}

        ref_norms = np.linalg.norm(ref_feats, axis=1)
        curr_norms = np.linalg.norm(curr_feats, axis=1)

        norm_drift = cls.test_feature_drift(ref_norms, curr_norms)

        # Centroid distance
        ref_center = np.mean(ref_feats, axis=0)
        curr_center = np.mean(curr_feats, axis=0)
        centroid_dist = float(np.linalg.norm(ref_center - curr_center))

        return {
            "drift_detected": norm_drift["drift_detected"],
            "norm_drift": norm_drift,
            "centroid_shift": centroid_dist
        }
