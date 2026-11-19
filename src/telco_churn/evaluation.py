
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)


THRESHOLD = 0.5


def evaluate_model(model, X_valid, y_valid):
    """Evaluate a fitted model on validation data."""

    # Locate the positive class (Churn = Yes)
    positive_index = list(model.classes_).index(1)

    positive_scores = model.predict_proba(X_valid)[:, positive_index]

    # Apply the fixed classification threshold
    predictions = (positive_scores >= THRESHOLD).astype(int)

    metrics = {
        "average_precision": average_precision_score(
            y_valid, positive_scores
        ),
        "accuracy": accuracy_score(y_valid, predictions),
        "precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_valid, predictions, zero_division=0
        ),
    }

    matrix = confusion_matrix(
        y_valid,
        predictions,
        labels=[0, 1],
    )

    metrics["tn"] = matrix[0, 0]
    metrics["fp"] = matrix[0, 1]
    metrics["fn"] = matrix[1, 0]
    metrics["tp"] = matrix[1, 1]

    return metrics
