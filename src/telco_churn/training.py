
import time

from telco_churn.pipeline import build_pipeline


def train_model(classifier, X_train, y_train):
    """Build and train a complete preprocessing and classification pipeline."""

    model = build_pipeline(classifier)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time

    return model, fit_time
