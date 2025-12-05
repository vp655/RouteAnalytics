# src/ravens_route/inference.py

from typing import Union
import numpy as np
import pandas as pd
import xgboost as xgb

from .models_io import get_route_model, get_route_features, get_route_encoder


def predict_route_prob(row: Union[pd.Series, pd.DataFrame]) -> float:
    """
    Compute P(catch | target, route-level features) for a single play.

    Assumes the input row is already preprocessed in the same way as the
    training data used for the route model (e.g., 'route' is already
    label-encoded, no extra columns in the feature matrix, etc.).

    Parameters
    ----------
    row : pd.Series or single-row pd.DataFrame
        Must contain at least the columns listed in route_features.json.

    Returns
    -------
    float
        Probability between 0 and 1.
    """
    model = get_route_model()
    features = get_route_features()

    row = row.copy()
    vals = row[features]

    encoder = get_route_encoder()
    route_str = vals["route"]
    vals["route"] = encoder[route_str]


    vals_numeric = pd.to_numeric(vals, errors="coerce").astype("float32")

    if vals_numeric.isna().any():
        nan_feats = list(vals_numeric[vals_numeric.isna()].index)
        raise ValueError(
            f"NaN encountered in features {nan_feats}. "
            "Make sure all model features are numeric / preprocessed."
        )

    # Shape (1, n_features) – mimics X_full_raw.values row
    X = vals_numeric.to_numpy(dtype="float32").reshape(1, -1)


    dmat = xgb.DMatrix(X, feature_names=features)
    proba = model.predict(dmat)[0]

    return float(proba)
