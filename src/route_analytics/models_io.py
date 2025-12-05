from pathlib import Path
import json
from typing import List, Optional, Dict

import xgboost as xgb

PACKAGE_DIR = Path(__file__).resolve().parent

models_dir = PACKAGE_DIR / "models"

if not models_dir.exists():
    repo_root = PACKAGE_DIR.parent.parent
    alt = repo_root / "models"
    if alt.exists():
        models_dir = alt

MODELS_DIR = models_dir

_route_model: Optional[xgb.Booster] = None
_route_features: Optional[List[str]] = None
_route_encoder: Optional[Dict[str, int]] = None


def get_route_features() -> List[str]:
    global _route_features

    if _route_features is None:
        features_path = MODELS_DIR / "route_features.json"
        if not features_path.exists():
            raise FileNotFoundError(f"Could not find route_features.json at {features_path}")

        with open(features_path, "r") as f:
            _route_features = json.load(f)

        if not isinstance(_route_features, list):
            raise ValueError("route_features.json must contain a JSON list of feature names.")

    return _route_features


def get_route_encoder() -> Dict[str, int]:
    global _route_encoder

    if _route_encoder is None:
        enc_path = MODELS_DIR / "route_label_mapping.json"
        if not enc_path.exists():
            raise FileNotFoundError(f"Could not find route_label_mapping.json at {enc_path}")

        with open(enc_path, "r") as f:
            _route_encoder = json.load(f)

        if not isinstance(_route_encoder, dict):
            raise ValueError("route_label_mapping.json must contain a JSON object {route: code}.")

    return _route_encoder


def get_route_model() -> xgb.Booster:
    global _route_model

    if _route_model is None:
        model_path = MODELS_DIR / "route_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find route_model.json at {model_path}")

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        _route_model = booster

    return _route_model
