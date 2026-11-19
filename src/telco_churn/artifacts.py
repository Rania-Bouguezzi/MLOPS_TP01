
from pathlib import Path
import joblib


MODEL_PATH = Path("models/churn_pipeline.joblib")


def save_model(model, path=MODEL_PATH):
    """Save the complete fitted pipeline."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    return path


def load_model(path=MODEL_PATH):
    """Load a previously saved fitted pipeline."""

    return joblib.load(path)
