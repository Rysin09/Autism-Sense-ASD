# 🧩 AutismSense — ASD Screening Tool

A machine-learning powered Streamlit web app for early Autism Spectrum Disorder (ASD) screening using the AQ-10 questionnaire and demographic data.

## Features

- **Interactive AQ-10 Screening** — 10-question validated instrument with real-time scoring
- **ML-Powered Prediction** — Random Forest classifier (82% test accuracy) with confidence scores
- **Visual Results** — Plotly gauge chart, AQ-10 score breakdown bar chart
- **Model Transparency** — Dedicated page explaining model performance, pipeline, and limitations
- **Educational Content** — ASD awareness, scoring logic, methodology explained

## Project Structure

```
autism_app/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── models/
│   ├── best_model.pkl      # Trained Random Forest (tuned via RandomizedSearchCV)
│   └── label_encoders.pkl  # LabelEncoders for categorical features
└── README.md
```

## Quick Start

```bash
# 1. Clone / navigate to the project folder
cd autism_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## Model Details

| Metric | Value |
|---|---|
| Algorithm | Random Forest |
| Hyperparameters | bootstrap=False, max_depth=20, n_estimators=50 |
| Test Accuracy | 82% |
| ASD Recall | 64% |
| ASD Precision | 59% |
| ASD F1-Score | 0.61 |
| Class Imbalance Handling | SMOTE |
| Hyperparameter Search | RandomizedSearchCV (5-fold CV, 20 iterations) |

## Features Used

**Behavioral (AQ-10 Scores):** A1_Score through A10_Score, result (aggregate)

**Demographic:** age, gender, ethnicity, jaundice, austim (family history), contry_of_res, used_app_before, relation

## Disclaimer

This tool is for **educational and screening purposes only**. It is **not a clinical diagnosis**. A score suggesting ASD traits should be followed up with a licensed healthcare professional.

---

Built with Python · scikit-learn · Streamlit · Plotly
