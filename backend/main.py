
import json
from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.database import get_connection, initialize_database


# --------------------------------------------------
# APPLICATION SETUP
# --------------------------------------------------

app = FastAPI(
    title="Parkinson's Digital Twin API",
    description="Prediction, history, statistics, and model evaluation API.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "results" / "parkinsons_random_forest.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Machine learning model loaded successfully.")
except Exception as exc:
    model = None
    print(f"Could not load machine learning model: {exc}")


# --------------------------------------------------
# MODEL FEATURE ORDER
# --------------------------------------------------

FEATURE_NAMES = [
    "MDVP_Fo_Hz",
    "MDVP_Fhi_Hz",
    "MDVP_Flo_Hz",
    "MDVP_Jitter_percent",
    "MDVP_Jitter_Abs",
    "MDVP_RAP",
    "MDVP_PPQ",
    "Jitter_DDP",
    "MDVP_Shimmer",
    "MDVP_Shimmer_dB",
    "Shimmer_APQ3",
    "Shimmer_APQ5",
    "MDVP_APQ",
    "Shimmer_DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class PredictionInput(BaseModel):
    MDVP_Fo_Hz: float = Field(..., allow_inf_nan=False)
    MDVP_Fhi_Hz: float = Field(..., allow_inf_nan=False)
    MDVP_Flo_Hz: float = Field(..., allow_inf_nan=False)
    MDVP_Jitter_percent: float = Field(..., allow_inf_nan=False)
    MDVP_Jitter_Abs: float = Field(..., allow_inf_nan=False)
    MDVP_RAP: float = Field(..., allow_inf_nan=False)
    MDVP_PPQ: float = Field(..., allow_inf_nan=False)
    Jitter_DDP: float = Field(..., allow_inf_nan=False)
    MDVP_Shimmer: float = Field(..., allow_inf_nan=False)
    MDVP_Shimmer_dB: float = Field(..., allow_inf_nan=False)
    Shimmer_APQ3: float = Field(..., allow_inf_nan=False)
    Shimmer_APQ5: float = Field(..., allow_inf_nan=False)
    MDVP_APQ: float = Field(..., allow_inf_nan=False)
    Shimmer_DDA: float = Field(..., allow_inf_nan=False)
    NHR: float = Field(..., allow_inf_nan=False)
    HNR: float = Field(..., allow_inf_nan=False)
    RPDE: float = Field(..., allow_inf_nan=False)
    DFA: float = Field(..., allow_inf_nan=False)
    spread1: float = Field(..., allow_inf_nan=False)
    spread2: float = Field(..., allow_inf_nan=False)
    D2: float = Field(..., allow_inf_nan=False)
    PPE: float = Field(..., allow_inf_nan=False)


class EvaluationInput(BaseModel):
    prediction_id: int = Field(..., gt=0)
    actual_label: int = Field(..., ge=0, le=1)


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

@app.on_event("startup")
def startup_event():
    initialize_database()


# --------------------------------------------------
# HELPER: CALCULATE CONFUSION MATRIX
# --------------------------------------------------

def calculate_evaluation_metrics():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                actual_label,
                predicted_label
            FROM evaluations
        """)

        rows = cursor.fetchall()

    finally:
        connection.close()

    tn = sum(
        1 for row in rows
        if row["actual_label"] == 0 and row["predicted_label"] == 0
    )

    fp = sum(
        1 for row in rows
        if row["actual_label"] == 0 and row["predicted_label"] == 1
    )

    fn = sum(
        1 for row in rows
        if row["actual_label"] == 1 and row["predicted_label"] == 0
    )

    tp = sum(
        1 for row in rows
        if row["actual_label"] == 1 and row["predicted_label"] == 1
    )

    total = len(rows)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "evaluated_predictions": total,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "matrix": [
            [tn, fp],
            [fn, tp],
        ],
        "labels": {
            "actual": ["No Parkinson's", "Parkinson's"],
            "predicted": ["No Parkinson's", "Parkinson's"],
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
        },
    }


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Parkinson's Digital Twin API is running.",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
        "history_endpoint": "/predictions",
        "statistics_endpoint": "/statistics",
        "evaluation_endpoint": "/evaluate",
        "confusion_matrix_endpoint": "/confusion-matrix",
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The trained machine learning model could not be loaded.",
        )

    try:
        input_values = data.model_dump()

        feature_values = [
            input_values[name]
            for name in FEATURE_NAMES
        ]

        features = np.array(
            [feature_values],
            dtype=float,
        )

        prediction = int(model.predict(features)[0])

        result = (
            "Parkinson's detected"
            if prediction == 1
            else "No Parkinson's detected"
        )

        connection = get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO predictions (input_data, prediction)
                VALUES (?, ?)
                """,
                (
                    json.dumps(input_values),
                    result,
                ),
            )

            prediction_id = cursor.lastrowid
            connection.commit()

        finally:
            connection.close()

        return {
            "id": prediction_id,
            "prediction": prediction,
            "result": result,
            "saved_to_database": True,
            "message": "Prediction completed and saved successfully.",
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# --------------------------------------------------
# PREDICTION HISTORY ENDPOINT
# --------------------------------------------------

@app.get("/predictions")
def get_predictions(
    limit: int = Query(default=20, ge=1, le=500)
):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, input_data, prediction, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        predictions = []

        for row in rows:
            predictions.append({
                "id": row["id"],
                "input_data": row["input_data"],
                "prediction": row["prediction"],
                "created_at": row["created_at"],
            })

        return {
            "count": len(predictions),
            "predictions": predictions,
        }

    finally:
        connection.close()


# --------------------------------------------------
# STATISTICS ENDPOINT
# --------------------------------------------------

@app.get("/statistics")
def get_statistics():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE prediction = ?
            """,
            ("Parkinson's detected",),
        )
        positive = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE prediction = ?
            """,
            ("No Parkinson's detected",),
        )
        negative = cursor.fetchone()[0]

        return {
            "total_predictions": total,
            "positive_predictions": positive,
            "negative_predictions": negative,
        }

    finally:
        connection.close()


# --------------------------------------------------
# SAVE ACTUAL LABEL AND EVALUATE PREDICTION
# --------------------------------------------------

@app.post("/evaluate")
def evaluate_prediction(data: EvaluationInput):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        # Find the saved prediction.
        cursor.execute(
            """
            SELECT id, prediction
            FROM predictions
            WHERE id = ?
            """,
            (data.prediction_id,),
        )

        prediction_row = cursor.fetchone()

        if prediction_row is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction ID was not found.",
            )

        # Convert the saved prediction text into its numeric label.
        saved_result = prediction_row["prediction"]

        if saved_result == "Parkinson's detected":
            predicted_label = 1
        elif saved_result == "No Parkinson's detected":
            predicted_label = 0
        else:
            raise HTTPException(
                status_code=400,
                detail="The saved prediction has an unrecognized label.",
            )

        # Do not evaluate the same prediction more than once.
        cursor.execute(
            """
            SELECT id
            FROM evaluations
            WHERE prediction_id = ?
            """,
            (data.prediction_id,),
        )

        if cursor.fetchone() is not None:
            raise HTTPException(
                status_code=409,
                detail="This prediction has already been evaluated.",
            )

        # Save the known actual label and the model's prediction.
        cursor.execute(
            """
            INSERT INTO evaluations (
                prediction_id,
                actual_label,
                predicted_label
            )
            VALUES (?, ?, ?)
            """,
            (
                data.prediction_id,
                data.actual_label,
                predicted_label,
            ),
        )

        connection.commit()

        metrics = calculate_evaluation_metrics()

        return {
            "message": "Prediction evaluated successfully.",
            "prediction_id": data.prediction_id,
            "actual_label": data.actual_label,
            "predicted_label": predicted_label,
            "correct_prediction": (
                data.actual_label == predicted_label
            ),
            "evaluation": metrics,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(exc)}",
        )

    finally:
        connection.close()


# --------------------------------------------------
# DYNAMIC CONFUSION MATRIX ENDPOINT
# --------------------------------------------------

@app.get("/confusion-matrix")
def get_confusion_matrix():
    return calculate_evaluation_metrics()


# --------------------------------------------------
# EVALUATION HISTORY ENDPOINT
# --------------------------------------------------

@app.get("/evaluations")
def get_evaluations(
    limit: int = Query(default=20, ge=1, le=500)
):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                e.id,
                e.prediction_id,
                e.actual_label,
                e.predicted_label,
                e.created_at,
                p.prediction
            FROM evaluations AS e
            LEFT JOIN predictions AS p
                ON e.prediction_id = p.id
            ORDER BY e.id DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        evaluations = [
            {
                "id": row["id"],
                "prediction_id": row["prediction_id"],
                "actual_label": row["actual_label"],
                "predicted_label": row["predicted_label"],
                "prediction": row["prediction"],
                "correct_prediction": (
                    row["actual_label"] == row["predicted_label"]
                ),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

        return {
            "count": len(evaluations),
            "evaluations": evaluations,
        }

    finally:
        connection.close()
