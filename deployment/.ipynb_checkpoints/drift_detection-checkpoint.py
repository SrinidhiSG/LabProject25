import json
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
reference_file_path=r"deployment/reference_profile.json"

from scipy.stats import ks_2samp, chi2_contingency
import pandas as pd
import json

def detect_drift(recent_df, reference_file_path=reference_file_path, threshold=0.3, feature_weights=None, pval_threshold=0.05):
    """
    Detects drift based on weighted contribution of all features.
    Returns: (drift_detected, drift_score, drift_flags)
    
    Parameters:
    - threshold: overall drift score threshold (0-1).
    - feature_weights: dict {feature: weight}, defaults to equal weights.
    - pval_threshold: per-feature significance level.
    """
    with open(reference_file_path, "r") as f:
        reference_profile = json.load(f)

    drift_flags = {}
    feature_scores = {}

    # If no weights provided, use equal weights
    if feature_weights is None:
        feature_weights = {f: 1.0 for f in reference_profile.keys()}
    # Normalize weights
    total_wt = sum(feature_weights.values())
    feature_weights = {f: w / total_wt for f, w in feature_weights.items()}

    for feature, stats in reference_profile.items():
        if feature not in recent_df:
            continue

        if stats["type"] == "num":
            recent_values = recent_df[feature].dropna()
            if len(recent_values) < 10:
                continue

            ref_mean = stats["mean"]
            ref_std = stats["std"]

            # Compare distributions: KS test
            ref_sample = pd.Series([ref_mean + ref_std for _ in range(len(recent_values))])
            _, p_val = ks_2samp(recent_values, ref_sample)

            drift_flags[feature] = (p_val < pval_threshold)
            feature_scores[feature] = 1 - p_val  # higher = more drift

        elif stats["type"] == "cat":
            ref_dist = stats["freq_dist"]
            recent_dist = recent_df[feature].value_counts(normalize=True, dropna=False).to_dict()

            all_categories = set(ref_dist.keys()) | set(recent_dist.keys())
            ref_counts = [ref_dist.get(cat, 0) for cat in all_categories]
            recent_counts = [recent_dist.get(cat, 0) for cat in all_categories]

            _, p_val, _, _ = chi2_contingency([ref_counts, recent_counts])

            drift_flags[feature] = (p_val < pval_threshold)
            feature_scores[feature] = 1 - p_val  # higher = more drift

    # Compute overall drift score as weighted sum
    drift_score = sum(feature_weights[f] * feature_scores.get(f, 0) for f in feature_weights)
    drift_detected = drift_score >= threshold

    return drift_detected, drift_score, drift_flags
