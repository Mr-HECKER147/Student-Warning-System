# Academic Risk Early Warning System 🚨

*Live Demo: https://student-warning-system-6wxtgukmjz23rmrl7jt99e.streamlit.app/*

## 🎯 Problem Statement

Many students fall behind academically before anyone notices. This system uses attendance, test scores, assignments, participation, and GPA to predict whether a student is **At Risk** or **Safe** — early enough for intervention.

## 📊 Key Results

| Metric | Value |
|---|---|
| **Model** | LightGBM |
| **Dataset** | 2000 students (synthetic) |
| **SMOTE** | Applied (class balancing) |
| **Engineered Features** | `test_avg`, `risk_score` |
| **Explainability** | SHAP values |

## 🛠 Tech Stack

- **Python 3.10+**
- **LightGBM** — gradient boosting classifier
- **Scikit-learn** — preprocessing, train/test split, cross-validation
- **imbalanced-learn** — SMOTE oversampling
- **SHAP** — feature explainability
- **Streamlit** — live UI
- **Pandas / NumPy / Matplotlib** — data and visualization

## 🚀 How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python generate_data.py

# 3. Train the model
python Model.py

# 4. Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
Student-Warning-System/
├── app.py                # Streamlit UI
├── Model.py              # LightGBM training + prediction
├── generate_data.py      # Synthetic dataset generator
├── requirements.txt      # Dependencies
├── README.md
├── data/                 # Generated dataset (git-ignored)
│   └── student_data.csv
└── model/                # Saved model artifacts (git-ignored)
    ├── risk_model.pkl
    ├── scaler.pkl
    └── model_info.json
```

## 🧠 Features Used

| Feature | Description |
|---|---|
| `internal_test1` | Internal test 1 score (/25) |
| `internal_test2` | Internal test 2 score (/25) |
| `attendance_pct` | Attendance percentage (0–100) |
| `assignments_avg` | Average assignment score (/20) |
| `participation` | Class participation score (1–5) |
| `prev_sem_gpa` | Previous semester GPA (0–10) |
| `test_avg` | *(engineered)* Mean of test1 + test2 |
| `risk_score` | *(engineered)* Weighted composite score |

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a pull request.

## 🎓 Credits

ViMEET · Foundations of AI VAP · Feb 2026
