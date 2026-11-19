
from sklearn.ensemble import RandomForestClassifier

from telco_churn.data import load_data, split_data
from telco_churn.pipeline import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from telco_churn.training import train_model
from telco_churn.evaluation import evaluate_model
from telco_churn.artifacts import save_model
from telco_churn.metadata import save_metadata


RANDOM_STATE = 42

def run_training():

    # 1. Load and split data
    df = load_data()
    train_df, valid_df, test_df = split_data(df)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train = train_df[features]
    y_train = train_df["Churn"].map({"No": 0, "Yes": 1})

    X_valid = valid_df[features]
    y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

    # 2. Define selected classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 3. Train complete pipeline
    model, fit_time = train_model(
        classifier,
        X_train,
        y_train,
    )

    # 4. Evaluate on validation set
    metrics = evaluate_model(model, X_valid, y_valid)

    # 5. Save complete fitted pipeline
    model_path = save_model(model)

    # 6. Save metadata
    metadata_path = save_metadata(
        metrics=metrics,
        classifier=classifier,
        train_ids=train_df["customerID"],
        valid_ids=valid_df["customerID"],
        test_ids=test_df["customerID"],
        random_state=RANDOM_STATE,
    )

    print("Training completed.")
    print("Fit time:", round(fit_time, 4), "seconds")
    print("Validation metrics:", metrics)
    print("Model saved to:", model_path)
    print("Metadata saved to:", metadata_path)


if __name__ == "__main__":
    run_training()
