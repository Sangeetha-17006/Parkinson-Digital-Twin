import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("dataset/parkinsons.data")

# Features and target
X = df.drop(columns=["name", "status"])
y = df["status"]


# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)


# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Evaluation metrics
scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]


results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring
)


print("5-Fold Cross-Validation Results")
print("--------------------------------")

print(f"Accuracy : {results['test_accuracy'].mean():.4f}")
print(f"Precision: {results['test_precision'].mean():.4f}")
print(f"Recall   : {results['test_recall'].mean():.4f}")
print(f"F1 Score : {results['test_f1'].mean():.4f}")

print()
print("Individual fold accuracies:")
print(results["test_accuracy"])