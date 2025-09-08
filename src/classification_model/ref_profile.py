import numpy as np

FEATURE_TYPES = {
    "Age": "num",
    "Fare": "num",
    "Pclass": "cat",
    "Sex": "cat",
    "SibSp": "cat",
    "Parch": "cat"
}

def build_reference_profile(df, feature_types=FEATURE_TYPES, num_bins=10):
    """
    Build a reference profile for numerical and categorical features.

    Parameters:
       df : pd.DataFrame
        Input dataframe.
    feature_types : dict
        Mapping of column names to feature types ("num" or "cat").
    num_bins : int
        Number of bins to use for histograms of numerical features.

    Returns:
    dict
        A dictionary with profiling statistics for each feature.
    """
    profile = {}

    for col, ftype in feature_types.items():
        if col not in df.columns:
            continue

        series = df[col].dropna()

        if ftype == "num":
            # histogram
            hist_counts, bin_edges = np.histogram(series, bins=num_bins, density=True)

            profile[col] = {
                "type": "num",
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "quantiles": {
                    q: float(series.quantile(q))
                    for q in [0.01, 0.25, 0.5, 0.75, 0.99]
                },
                "hist_counts": hist_counts.tolist(),
                "hist_bins": bin_edges.tolist()
            }

        elif ftype == "cat":
            freq_dist = (
                series.value_counts(normalize=True, dropna=False)
                .astype(float)
                .to_dict()
            )
            profile[col] = {
                "type": "cat",
                "freq_dist": {str(k): v for k, v in freq_dist.items()}
            }

    return profile
