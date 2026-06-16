<div align="center">

# 🧩 AutismSense
### ML-Powered ASD Early Screening Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=flat-square&logo=streamlit)](https://your-app-url.streamlit.app)

<p align="center">
  An end-to-end machine learning web application for Autism Spectrum Disorder (ASD) early screening,<br/>
  built on the clinically validated AQ-10 instrument with a Random Forest classifier tuned via RandomizedSearchCV.
</p>

---

</div>

## 📌 Overview

AutismSense is a full-stack ML screening tool that operationalizes a trained Random Forest model into a production-ready web interface. Users complete the AQ-10 behavioral questionnaire alongside demographic inputs; the model returns a real-time ASD probability with visual explainability.

The project demonstrates an end-to-end ML engineering workflow — from raw data preprocessing and class imbalance handling to hyperparameter optimization, model serialization, and interactive deployment.

> ⚠️ **Medical Disclaimer:** This tool is strictly for educational and research purposes. It does not constitute a clinical diagnosis. Any positive screening result should be followed up with a licensed healthcare professional.

---

## 📌 Live Demo

👉 **[Launch App →](https://your-app-url.streamlit.app)**

---

## 📌 Features

| Feature | Description |
|---|---|
| **AQ-10 Screener** | Clinically validated 10-item behavioral questionnaire with reverse scoring |
| **ML Inference** | Random Forest with probability calibration and confidence display |
| **Visual Explainability** | Plotly gauge chart for ASD probability + per-question AQ score breakdown |
| **Model Transparency** | Dedicated insights page with CV metrics, pipeline architecture, and feature importances |
| **Responsive UI** | Dark-mode premium design (Syne + DM Sans) with custom Streamlit CSS overrides |
| **Caching** | `@st.cache_resource` for zero-latency model loading after first run |

---

## 📌 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                  │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────┐  │
│  │  Screening Tool │  │ Model Insights │  │How It    │  │
│  │  (AQ-10 Form)   │  │ (CV Metrics)   │  │Works     │  │
│  └────────┬────────┘  └────────────────┘  └──────────┘  │
│           │ 18 features                                 │
│  ┌────────▼────────────────────────────────────────┐    │
│  │              Inference Pipeline                 │    │
│  │  LabelEncoders → Feature Array → RF Classifier  │    │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

---

## 📌 ML Pipeline

```
Raw Data (Kaggle ASD Dataset)
        │
        ▼
Data Preprocessing
  ├── Drop low-variance features (age_desc)
  ├── Standardize country names
  ├── Impute missing values (ethnicity, relation)
  └── LabelEncode categorical columns
        │
        ▼
Train / Test Split  (80/20, stratified)
        │
        ▼
SMOTE Oversampling  (minority class balancing on train set only)
        │
        ▼
RandomizedSearchCV  (5-fold CV, 20 iterations, scoring=F1)
  ├── DecisionTreeClassifier
  ├── RandomForestClassifier  ← best
  └── XGBClassifier
        │
        ▼
Final Evaluation on held-out test set
        │
        ▼
Model Serialization  (pickle → best_model.pkl)
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest |
| Best Hyperparameters | `bootstrap=False, max_depth=20, n_estimators=50` |
| Hyperparameter Search | RandomizedSearchCV · 5-fold CV · 20 iterations |
| Optimization Metric | F1-Score (minority class aware) |
| Class Imbalance | SMOTE on training set |
| **Test Accuracy** | **82%** |
| ASD Precision | 59% |
| ASD Recall | 64% |
| ASD F1-Score | 0.61 |

> Recall is prioritized over precision in this domain — missing a true ASD case (false negative) carries higher real-world cost than a false positive.

### Feature Importances (Top 5)

| Rank | Feature | Importance |
|---|---|---|
| 1 | A9_Score | 0.121 |
| 2 | age | 0.101 |
| 3 | ethnicity | 0.098 |
| 4 | contry_of_res | 0.084 |
| 5 | A3_Score | 0.078 |

---

## 🗂️ Project Structure

```
Autism-Sense-ASD/
│
├── app.py                        # Streamlit application (3-page multi-view)
├── requirements.txt              # Pinned dependencies
├── README.md
│
└── models/
    ├── best_model.pkl            # Serialized Random Forest (sklearn 1.6.1)
    └── label_encoders.pkl        # Fitted LabelEncoders for 7 categorical features
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Rysin09/Autism-Sense-ASD.git
cd Autism-Sense-ASD

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App will be available at `http://localhost:8501`

---

## 📌 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn 1.6.1 |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Boosting Baseline | XGBoost |
| Web Framework | Streamlit |
| Visualization | Plotly |
| Data | pandas, NumPy |
| Serialization | pickle |

---

## 📌 Roadmap

- [ ] SHAP explainability — per-prediction feature contribution waterfall chart
- [ ] Batch prediction — CSV upload for screening multiple subjects
- [ ] Model retraining UI — upload new data and trigger pipeline from the app
- [ ] FastAPI backend — decouple inference from the Streamlit frontend
- [ ] Docker containerization — `docker-compose up` single-command deploy

---

## 👤 Author

**Aryan** — B.Sc. (Hons) Computer Science & Data Analytics, IIT Patna

[![GitHub](https://img.shields.io/badge/GitHub-Rysin09-181717?style=flat-square&logo=github)](https://github.com/Rysin09)

---

<div align="center">
  <sub>Built with Python · scikit-learn · Streamlit · Plotly &nbsp;|&nbsp; For educational use only</sub>
</div>
