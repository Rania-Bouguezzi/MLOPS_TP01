
import json
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version

from telco_churn.pipeline import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

METADATA_PATH = Path("models/metadata.json")


def calculate_file_hash(path):
    """Calculate the SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def save_metadata(
    metrics,
    classifier,
    train_ids,
    valid_ids,
    test_ids,
    data_path="data/raw/telco.csv",
    random_state=42,
):
    """Save model configuration and training metadata."""

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": classifier.__class__.__name__,
        "model_parameters": classifier.get_params(),
        "primary_metric": "average_precision",
        "validation_metrics": {
            key: float(value) for key, value in metrics.items()
        },
        "positive_class": 1,
        "classification_threshold": 0.5,
        "random_state": random_state,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "split": {
        "train_customer_ids": list(train_ids),
        "validation_customer_ids": list(valid_ids),
        "test_customer_ids": list(test_ids),
        },
        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "StandardScaler",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        },
        "data": {
            "path": data_path,
            "sha256": calculate_file_hash(data_path),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "joblib": version("joblib"),
        },
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    return METADATA_PATH
