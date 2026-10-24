import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("dataset/parkinsons.data")

# Features and target
X = df.drop(columns=["name", "status"])
y = df["status"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        class_weight="balanced",
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


# Train and evaluate
for name, model in models.items():

    if name == "Random Forest":
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    else:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print(classification_report(y_test, predictions))