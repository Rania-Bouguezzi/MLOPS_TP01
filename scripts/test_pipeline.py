
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from telco_churn.pipeline import build_pipeline


RANDOM_STATE = 42
DATA_PATH = "data/raw/telco.csv"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Clean TotalCharges
    df["TotalCharges"] = df["TotalCharges"].replace(
        r"^\s*$", pd.NA, regex=True
    )
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="raise"
    )

    # Prepare target
    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df[FEATURES].copy()

    # Train / validation / test split
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=RANDOM_STATE
    )

    # Build and fit pipeline
    model = build_pipeline()
    model.fit(X_train, y_train)

    # Validation predictions
    predictions = model.predict(X_valid)

    print("Validation accuracy:", accuracy_score(y_valid, predictions))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_valid, predictions, labels=[0, 1]))

    # Check transformed data dimensions
    X_transformed = model.named_steps["preprocessor"].transform(X_train)

    print("\nOriginal features:", X_train.shape[1])
    print("Transformed features:", X_transformed.shape[1])

    print("\nPipeline test completed successfully.")


if __name__ == "__main__":
    main()
