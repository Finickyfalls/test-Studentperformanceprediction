# 🎓 Student Performance Prediction (SPP) 
A HOME WORK FOR IHIFIX ACADEMY VIA ubefu.
A machine learning web app that predicts a student's total score based on their weekly self-study hours, attendance percentage, and class participation.

**🔗 Live App:** https://test-studentperformanceprediction-cltyprbznkk7ibqslzbhtp.streamlit.app/

## Overview

This project applies a full end-to-end ML workflow: data cleaning, exploratory data analysis, model training and evaluation, model serialization, and deployment as an interactive Streamlit web app.

## Dataset

- Source: Kaggle — Student Performance dataset
- Original size: ~1,000,000 rows, trimmed to ~50,000 rows for this project
- Link; NOTE: Original csv dataset and the adjusted trimmed version are available on the repo.

**Features used:**
| Feature | Description |
|---|---|
| `weekly_self_study_hours` | Hours per week the student spends studying independently |
| `attendance_percentage` | Class attendance rate (%) |
| `class_participation` | Class participation score (0–10) |

**Target:** `total_score` (0–100)

> Note: the dataset also included a `grade` column (A–F), but it was excluded from training since it's directly derived from `total_score` = including it would leak the answer into the model.

## Project Workflow

1. **Data Cleaning** — checked for nulls, duplicates, and inspected categorical/numerical columns
2. **EDA** — distribution plots, boxplots for outliers, correlation heatmap
3. **Preprocessing** — outlier removal (IQR method), StandardScaler on numeric features
4. **Model Training** — compared 8 regression models (Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, XGBoost, LightGBM) using `RandomizedSearchCV` with 5-fold cross-validation
5. **Evaluation** — best model selected by R² score, evaluated on a held-out test set with MAE, RMSE, and R²
6. **Serialization** — best model and scaler saved with `joblib`
7. **Deployment** — Streamlit app built and deployed on Streamlit Community Cloud

## Repository Contents

| File | Description |
|---|---|
| `HW_Student_Preformance_Prediction_SPP.ipynb` | Full notebook: cleaning, EDA, training, evaluation |
| `Adjusted Student performance.csv` | Dataset used for training |
| `Original Student Performance Dataset .csv` | Dataset |
| `app.py` | Streamlit application |
| `spp_best_model.joblib` | Serialized best-performing model |
| `spp_scaler.joblib` | Serialized StandardScaler fitted on training data |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Pinned Python version for deployment |
| `README.md` | This file |

## Running Locally

```bash
git clone (https://github.com/Finickyfalls/test-Studentperformanceprediction)
cd     test-Studentperformanceprediction
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, xgboost, lightgbm, joblib, streamlit
- **Deployment:** Streamlit Community Cloud

## Author

Finicky Falls
