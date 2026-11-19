
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

from telco_churn.pipeline import build_pipeline


# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_STATE = 42
DATA_PATH = "data/raw/telco.csv"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

THRESHOLD = 0.5


# ==========================================
# 2. LOAD AND PREPARE DATA
# ==========================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    # Validate required columns
    required_columns = [
        "customerID", "Churn"
    ] + FEATURES

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Normalize whitespace-only TotalCharges
    df["TotalCharges"] = df["TotalCharges"].replace(
        r"^\s*$", pd.NA, regex=True
    )

    # Strict numeric conversion
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected values in Churn.")

    return df


# ==========================================
# 3. REPRODUCIBLE DATA SPLIT
# ==========================================

def split_data(df):

    train_df, remaining_df = train_test_split(
        df,
        test_size=0.40,
        stratify=df["Churn"],
        random_state=RANDOM_STATE,
    )

    valid_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        stratify=remaining_df["Churn"],
        random_state=RANDOM_STATE,
    )

    return train_df, valid_df, test_df


# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X_valid, y_valid):

    # Find the probability column for positive class 1
    positive_index = list(model.classes_).index(1)

    positive_scores = model.predict_proba(X_valid)[:, positive_index]

    # Apply the fixed threshold
    predictions = (positive_scores >= THRESHOLD).astype(int)

    # Calculate metrics
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


# ==========================================
# 5. MODEL COMPARISON
# ==========================================

def main():

    df = load_data()

    train_df, valid_df, test_df = split_data(df)

    # Check customer ID separation
    assert set(train_df["customerID"]).isdisjoint(valid_df["customerID"])
    assert set(train_df["customerID"]).isdisjoint(test_df["customerID"])
    assert set(valid_df["customerID"]).isdisjoint(test_df["customerID"])

    # Build features and target
    X_train = train_df[FEATURES].copy()
    y_train = train_df["Churn"].map({"No": 0, "Yes": 1})

    X_valid = valid_df[FEATURES].copy()
    y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})

    print("Train size:", len(X_train))
    print("Validation size:", len(X_valid))
    print("Test size:", len(test_df))
    print("Test set is reserved and will not be evaluated.")

    # Three learned classifiers
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []

    # Train and evaluate each learned classifier
    for name, classifier in models.items():

        print(f"\nTraining {name}...")

        candidate = build_pipeline(classifier)

        start_time = time.perf_counter()
        candidate.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_time

        metrics = evaluate_model(candidate, X_valid, y_valid)

        results.append({
            "model": name,
            "configuration": str(classifier),
            "fit_time_seconds": fit_time,
            **metrics,
        })

        print("Average Precision:", round(metrics["average_precision"], 4))
        print("Accuracy:", round(metrics["accuracy"], 4))
        print("Precision:", round(metrics["precision"], 4))
        print("Recall:", round(metrics["recall"], 4))
        print("F1:", round(metrics["f1"], 4))
        print("Fit time:", round(fit_time, 4), "seconds")

    # Majority baseline
    print("\nTraining Majority Baseline...")

    baseline = DummyClassifier(strategy="most_frequent")

    start_time = time.perf_counter()
    baseline.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time

    baseline_metrics = evaluate_model(
        baseline, X_valid, y_valid
    )

    results.append({
        "model": "Majority Baseline",
        "configuration": "DummyClassifier(strategy='most_frequent')",
        "fit_time_seconds": fit_time,
        **baseline_metrics,
    })

    # Save comparison table
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="average_precision",
        ascending=False,
    )

    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        "reports/comparison_results.csv",
        index=False,
    )

    # Save selection note
    selected_model = results_df.iloc[0]["model"]

    selection_note = f"""
Model selection note

Primary metric: Average Precision
Reason: Churn = Yes is the minority positive class.
Selection rule: Highest validation Average Precision.

Selected model: {selected_model}

Threshold for precision, recall and F1: {THRESHOLD}

The comparison uses the same six features and the same
stratified train/validation/test split for every candidate.

The test set was reserved and was not used for model selection.
"""

    with open("reports/selection_note.txt", "w", encoding="utf-8") as file:
        file.write(selection_note)

    print("\nSelected model:", selected_model)
    print("\nComparison saved to reports/comparison_results.csv")
    print("Selection note saved to reports/selection_note.txt")


if __name__ == "__main__":
    main()
