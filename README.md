# Telco Churn Prediction — MLOps TP02

## Project description

This project predicts customer churn using a reusable Scikit-learn training pipeline.

The target variable is `Churn`:

- `0`: No churn
- `1`: Churn

## Installation

Install dependencies with:

```bash
uv sync
```

## Data

Place the original dataset at:

`data/raw/telco.csv`

The dataset is split into:

- 60% training
- 20% validation
- 20% testing

The split is stratified and uses random state 42.

`customerID` is used only for identification and split verification. It is not used as a model feature.

## Features

Numeric:

- tenure
- MonthlyCharges
- TotalCharges

Categorical:

- Contract
- InternetService
- PaymentMethod

## Training

Run the complete training workflow:

```bash
uv run python -m telco_churn.workflow
```

The workflow trains the selected Random Forest pipeline, evaluates it on the validation set, and saves the model and metadata.

## Saved artifacts

- `models/churn_pipeline.joblib`: fitted preprocessing and classification pipeline.
- `models/metadata.json`: model parameters, validation metrics, split information, data hash, and environment information.
- `reports/comparison_results.csv`: model comparison results.
- `reports/selection_note.txt`: model selection rationale.

## Prediction example

```python
import pandas as pd
from telco_churn.artifacts import load_model

model = load_model()

X_new = pd.DataFrame([{
    "tenure": 12,
    "MonthlyCharges": 70,
    "TotalCharges": 840,
    "Contract": "Month-to-month",
    "InternetService": "DSL",
    "PaymentMethod": "Electronic check"
}])

prediction = model.predict(X_new)
probability = model.predict_proba(X_new)[:, 1]

print(prediction)
print(probability)
```

## Preprocessing

- Missing numeric values are replaced with the training median.
- Numeric features are standardized.
- Missing categorical values are replaced with the most frequent training value.
- Categorical features are one-hot encoded.
- Unknown categories are ignored during prediction.

The fitted preprocessing steps are saved together with the classifier.
