# 🧠 Parkinson's Digital Twin

A machine learning-powered web application that analyzes voice measurement features to predict Parkinson's disease status using a Random Forest classifier.

## 🌐 Live Demo

* **Live Application:** https://parkinson-digital-twin.netlify.app/
* **Backend API:** https://parkinson-digital-twin.onrender.com/
* **API Documentation:** https://parkinson-digital-twin.onrender.com/docs
* **GitHub Repository:** https://github.com/Sangeetha-17006/Parkinson-Digital-Twin

> Note: The application is an educational prototype, not a clinically validated diagnostic system.

## 📌 Project Overview

Parkinson's Digital Twin demonstrates how machine learning can be integrated with a web application to analyze voice measurement features and generate predictions.

The system combines a trained Random Forest model, a FastAPI backend, a SQLite database, and a responsive web dashboard.

## ✨ Features

* Voice measurement-based prediction using 22 input features
* Random Forest machine learning model
* Interactive web dashboard
* REST API built with FastAPI
* Prediction history stored in a database
* Dashboard statistics
* Confusion matrix visualization
* Model evaluation using accuracy, precision, recall, and F1 score
* Deployed frontend and backend

## 🛠️ Technologies Used

| Technology            | Purpose                           |
| --------------------- | --------------------------------- |
| Python                | Backend and machine learning      |
| Pandas, NumPy         | Data processing                   |
| Scikit-learn          | Model training and evaluation     |
| FastAPI               | REST API                          |
| SQLite                | Prediction and evaluation storage |
| HTML, CSS, JavaScript | Frontend dashboard                |
| Git and GitHub        | Version control                   |
| Render                | Backend deployment                |
| Netlify               | Frontend deployment               |

## 🏗️ System Architecture

1. **Frontend:** Collects voice measurement values and displays predictions.
2. **Backend:** FastAPI receives requests and processes prediction and evaluation operations.
3. **Machine Learning:** The trained Random Forest classifier generates predictions.
4. **Database:** SQLite stores prediction history and evaluation records.
5. **Dashboard:** Displays statistics, prediction history, confusion matrix, and evaluation metrics.

## 📂 Project Structure

```text
Parkinson-Digital-Twin/
├── backend/
│   ├── main.py
│   └── database.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── confusion_matrix.png
├── ml/
│   ├── train_model.py
│   ├── train_final_model.py
│   ├── compare_models.py
│   ├── cross_validate.py
│   └── evaluate_model.py
├── dataset/
├── docs/
├── notebooks/
├── results/
│   ├── confusion_matrix.png
│   └── parkinsons_random_forest.pkl
└── requirements.txt
```

*Note: The structure above summarizes the main project files; some local files may not be included in the public repository.*

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Sangeetha-17006/Parkinson-Digital-Twin.git
cd Parkinson-Digital-Twin
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start the backend

Run this command from the project root directory:

```bash
uvicorn backend.main:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

### 6. Open the frontend

Open `frontend/index.html` in a browser.

The frontend's API URL must point to the backend instance you intend to use. For local testing, configure it to use `http://127.0.0.1:8000`. For the deployed application, use the deployed Render API URL.

## 📊 Model Evaluation

The project includes scripts for model training, model comparison, cross-validation, and performance evaluation.

The local evaluation script previously reported:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 94.87% |
| Precision | 96.55% |
| Recall    | 96.55% |
| F1 Score  | 96.55% |

These results reflect a particular evaluation run and are not a guarantee of performance on unseen clinical data.

The live dashboard calculates evaluation metrics from saved prediction evaluations. Its metrics may be misleading when only a small number of predictions have been evaluated. Meaningful evaluation requires verified actual labels.

## ⚠️ Dataset and Disclaimer

Before redistributing the dataset, confirm its original source, license, attribution requirements, and permitted use.

This project is intended for educational and research purposes only. It is not a medical diagnostic tool, and its predictions must not be used as a substitute for professional medical advice or clinical diagnosis.

## 👩‍💻 Author

**Sangeetha S**

GitHub: https://github.com/Sangeetha-17006
