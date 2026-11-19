
import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn.pipeline import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

RANDOM_STATE = 42
DATA_PATH = "data/raw/telco.csv"

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path=DATA_PATH):
    """Load and validate the Telco dataset."""

    df = pd.read_csv(path)

    required_columns = ["customerID", "Churn"] + FEATURES
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Convert whitespace-only TotalCharges to missing values
    df["TotalCharges"] = df["TotalCharges"].replace(
        r"^\s*$", pd.NA, regex=True
    )

    # Strict numeric conversion
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected values in Churn.")

    return df


def split_data(df):
    """Create reproducible stratified train, validation and test sets."""

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

    # Verify that customer IDs do not overlap
    assert set(train_df["customerID"]).isdisjoint(valid_df["customerID"])
    assert set(train_df["customerID"]).isdisjoint(test_df["customerID"])
    assert set(valid_df["customerID"]).isdisjoint(test_df["customerID"])

    return train_df, valid_df, test_df
