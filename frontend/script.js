
const API_URL = "http://127.0.0.1:8000";

const FEATURE_NAMES = [
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
    "PPE"
];

document.addEventListener("DOMContentLoaded", () => {
    const predictionForm = document.getElementById("predictionForm");
    const predictButton = document.getElementById("predictButton");
    const resetButton = document.getElementById("resetButton");
    const predictionResult = document.getElementById("predictionResult");

    // Load dashboard data when the page opens.
    loadStatistics();
    loadPredictionHistory();
    loadConfusionMatrix();

    // Refresh buttons.
    document.getElementById("refreshHistoryButton")
        ?.addEventListener("click", loadPredictionHistory);

    document.getElementById("refreshEvaluationButton")
        ?.addEventListener("click", loadConfusionMatrix);

    // Handle prediction form submission.
    predictionForm?.addEventListener("submit", async (event) => {
        event.preventDefault();

        const formData = new FormData(predictionForm);
        const payload = {};

        for (const name of FEATURE_NAMES) {
            const rawValue = formData.get(name);

            if (rawValue === null || rawValue.trim() === "") {
                showPredictionResult(
                    "Please enter all 22 measurement values.",
                    false
                );
                return;
            }

            const numberValue = Number(rawValue);

            if (!Number.isFinite(numberValue)) {
                showPredictionResult(
                    `Please enter a valid number for ${name}.`,
                    false
                );
                return;
            }

            payload[name] = numberValue;
        }

        predictButton.disabled = true;
        predictButton.textContent = "Predicting...";

        showPredictionResult("Processing prediction...", true);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(
                    result.detail || "Prediction request failed."
                );
            }

            const message =
                `Prediction ID: ${result.id}\n` +
                `Result: ${result.result}\n\n` +
                "This is a machine-learning output, not a medical diagnosis.";

            showPredictionResult(message, true);

            // Refresh dashboard information after saving the prediction.
            await Promise.all([
                loadStatistics(),
                loadPredictionHistory()
            ]);

        } catch (error) {
            showPredictionResult(
                `Error: ${error.message}. Check that the backend is running.`,
                false
            );
        } finally {
            predictButton.disabled = false;
            predictButton.textContent = "Predict";
        }
    });

    // Clear prediction output when the form is reset.
    resetButton?.addEventListener("click", () => {
        predictionResult.hidden = true;
        predictionResult.textContent = "";
        predictionResult.classList.remove("success", "error");
    });
});


// --------------------------------------------------
// DISPLAY PREDICTION RESULT
// --------------------------------------------------

function showPredictionResult(message, success) {
    const element = document.getElementById("predictionResult");

    if (!element) return;

    element.textContent = message;
    element.hidden = false;

    element.classList.toggle("success", success);
    element.classList.toggle("error", !success);
}


// --------------------------------------------------
// LOAD DASHBOARD STATISTICS
// --------------------------------------------------

async function loadStatistics() {
    const status = document.getElementById("statisticsStatus");

    try {
        const response = await fetch(`${API_URL}/statistics`);

        if (!response.ok) {
            throw new Error("Could not load statistics.");
        }

        const data = await response.json();

        setText("totalPredictions", data.total_predictions ?? 0);
        setText("positivePredictions", data.positive_predictions ?? 0);
        setText("negativePredictions", data.negative_predictions ?? 0);

        if (status) {
            status.textContent = "Statistics updated successfully.";
        }

    } catch (error) {
        if (status) {
            status.textContent =
                "Unable to load statistics. Check the backend connection.";
        }

        console.error("Statistics error:", error);
    }
}


// --------------------------------------------------
// LOAD PREDICTION HISTORY
// --------------------------------------------------

async function loadPredictionHistory() {
    const status = document.getElementById("historyStatus");
    const tableBody = document.getElementById("historyTableBody");

    if (!tableBody) return;

    try {
        if (status) {
            status.textContent = "Loading prediction history...";
        }

        const response = await fetch(`${API_URL}/predictions?limit=20`);

        if (!response.ok) {
            throw new Error("Could not load prediction history.");
        }

        const data = await response.json();

        tableBody.replaceChildren();

        const predictions = data.predictions ?? [];

        if (predictions.length === 0) {
            const row = tableBody.insertRow();
            const cell = row.insertCell();

            cell.colSpan = 3;
            cell.textContent = "No predictions available yet.";

        } else {
            for (const prediction of predictions) {
                const row = tableBody.insertRow();

                const idCell = row.insertCell();
                const resultCell = row.insertCell();
                const dateCell = row.insertCell();

                idCell.textContent = prediction.id;
                resultCell.textContent = prediction.prediction;
                dateCell.textContent = prediction.created_at ?? "—";

                resultCell.classList.add(
                    prediction.prediction === "Parkinson's detected"
                        ? "positive-result"
                        : "negative-result"
                );
            }
        }

        if (status) {
            status.textContent =
                `${predictions.length} recent prediction(s) loaded.`;
        }

    } catch (error) {
        if (status) {
            status.textContent =
                "Unable to load prediction history.";
        }

        tableBody.replaceChildren();

        const row = tableBody.insertRow();
        const cell = row.insertCell();

        cell.colSpan = 3;
        cell.textContent = "Could not connect to the backend.";

        console.error("History error:", error);
    }
}


// --------------------------------------------------
// LOAD DYNAMIC CONFUSION MATRIX AND METRICS
// --------------------------------------------------

async function loadConfusionMatrix() {
    const status = document.getElementById("matrixStatus");

    try {
        if (status) {
            status.textContent = "Updating evaluation results...";
        }

        const response = await fetch(`${API_URL}/confusion-matrix`);

        if (!response.ok) {
            throw new Error("Could not load evaluation results.");
        }

        const data = await response.json();
        const matrix = data.confusion_matrix;
        const metrics = data.metrics;

        setText("tnCount", matrix.true_negative);
        setText("fpCount", matrix.false_positive);
        setText("fnCount", matrix.false_negative);
        setText("tpCount", matrix.true_positive);

        setText("accuracyValue", formatPercentage(metrics.accuracy));
        setText("precisionValue", formatPercentage(metrics.precision));
        setText("recallValue", formatPercentage(metrics.recall));
        setText("f1Value", formatPercentage(metrics.f1_score));

        setText(
            "evaluatedCount",
            `Evaluated predictions: ${data.evaluated_predictions}`
        );

        if (status) {
            status.textContent =
                data.evaluated_predictions === 0
                    ? "No evaluated predictions yet. Save an evaluation to populate the matrix."
                    : "Confusion matrix and metrics updated successfully.";
        }

    } catch (error) {
        if (status) {
            status.textContent =
                "Unable to load the confusion matrix. Check the backend connection.";
        }

        console.error("Evaluation error:", error);
    }
}


// --------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------

function setText(elementId, value) {
    const element = document.getElementById(elementId);

    if (element) {
        element.textContent = String(value);
    }
}


function formatPercentage(value) {
    if (!Number.isFinite(Number(value))) {
        return "—";
    }

    return `${(Number(value) * 100).toFixed(2)}%`;
}

