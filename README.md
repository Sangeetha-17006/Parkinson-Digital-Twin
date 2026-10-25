# 🧠 Parkinson's Digital Twin

A machine learning-based web application that predicts Parkinson's disease risk using voice measurement features.

## 📌 Project Overview

The Parkinson's Digital Twin project uses machine learning to analyze voice measurements and predict whether a person may have Parkinson's disease.

The application combines a machine learning model, a FastAPI backend, and a web-based frontend to provide predictions.

## ✨ Features

* Voice measurement-based prediction
* Random Forest machine learning model
* FastAPI backend
* Interactive web dashboard
* Model performance evaluation
* Confusion matrix visualization
* Prediction history and database integration

## 🛠️ Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* FastAPI
* SQLite
* HTML, CSS, JavaScript
* Git and GitHub

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
└── results/
    ├── confusion_matrix.png
    └── parkinsons_random_forest.pkl
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Sangeetha-17006/Parkinson-Digital-Twin.git
cd Parkinson-Digital-Twin
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn joblib fastapi uvicorn
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload
```

Open the API documentation at:

http://127.0.0.1:8000/docs

### 6. Open the frontend

Open `frontend/index.html` in your browser while the backend is running.

## 📊 Model Evaluation

The project includes scripts for model training, model comparison, cross-validation, and performance evaluation.

Evaluation results may vary depending on the training configuration and data split.

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool. Predictions should not be used as a substitute for professional medical advice or clinical diagnosis.

## 👩‍💻 Author

** Sangeetha S **

GitHub: [Sangeetha-17006](https://github.com/Sangeetha-17006)
