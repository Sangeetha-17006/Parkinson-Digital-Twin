import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("dataset/parkinsons.data")

# Features and target
X = df.drop(columns=["name", "status"])
y = df["status"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Final Random Forest model trained!")
print(f"Accuracy: {accuracy:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "results/parkinsons_random_forest.pkl")

print()
print("Model saved to:")
print("results/parkinsons_random_forest.pkl")