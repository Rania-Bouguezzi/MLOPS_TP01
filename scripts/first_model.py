
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_STATE = 42
DATA_PATH = "data/raw/telco.csv"

FEATURES = ["tenure", "MonthlyCharges"]


# ==========================================
# 2. CHARGEMENT ET NETTOYAGE
# ==========================================

def load_data(path):

    df = pd.read_csv(path)

    # Vérifier les colonnes avant toute manipulation
    required_columns = [
        "customerID", "tenure", "MonthlyCharges",
        "TotalCharges", "Churn"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    # Remplacer les chaînes vides ou composées d'espaces par NA
    df["TotalCharges"] = df["TotalCharges"].replace(
        r"^\s*$", pd.NA, regex=True
    )

    # Conversion numérique stricte
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="raise"
    )

    # Vérifier les valeurs de la cible
    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Valeurs inattendues dans la colonne Churn.")

    return df


# ==========================================
# 3. SPLIT TRAIN / VALIDATION / TEST
# ==========================================

def split_data(df):

    train_df, remaining_df = train_test_split(
        df,
        test_size=0.40,
        stratify=df["Churn"],
        random_state=RANDOM_STATE
    )

    valid_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        stratify=remaining_df["Churn"],
        random_state=RANDOM_STATE
    )

    return train_df, valid_df, test_df


# ==========================================
# 4. PREPARATION DES DONNEES
# ==========================================

def prepare_xy(df):

    X = df[FEATURES].copy()

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    if y.isna().any():
        raise ValueError("La cible contient des valeurs non reconnues.")

    return X, y


# ==========================================
# 5. EVALUATION
# ==========================================

def evaluate_model(name, model, X, y):

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy:.4f}")

    print("Confusion matrix:")
    print(matrix)

    print("Ordre des classes : [0 = No, 1 = Yes]")

    return accuracy


# ==========================================
# 6. EXECUTION
# ==========================================

def main():

    df = load_data(DATA_PATH)

    print("Dataset dimensions:", df.shape)

    train_df, valid_df, test_df = split_data(df)

    print("\nRépartition des données:")
    print("Train:", len(train_df))
    print("Validation:", len(valid_df))
    print("Test:", len(test_df))

    # Vérification des identifiants
    assert set(train_df["customerID"]).isdisjoint(
        valid_df["customerID"]
    )

    assert set(train_df["customerID"]).isdisjoint(
        test_df["customerID"]
    )

    assert set(valid_df["customerID"]).isdisjoint(
        test_df["customerID"]
    )

    print("\nAucun chevauchement entre les ensembles.")

    X_train, y_train = prepare_xy(train_df)
    X_valid, y_valid = prepare_xy(valid_df)

    # Modèle appris
    logistic_model = LogisticRegression(max_iter=1000)

    logistic_model.fit(X_train, y_train)

    evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_valid,
        y_valid
    )

    # Baseline
    baseline = DummyClassifier(strategy="most_frequent")

    baseline.fit(X_train, y_train)

    evaluate_model(
        "Dummy Classifier",
        baseline,
        X_valid,
        y_valid
    )


if __name__ == "__main__":
    main()
