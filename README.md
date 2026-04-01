# Academic Risk Early Warning System

*Live Demo: `streamlit run app.py`*

*Link: https://student-warning-system-6wxtgukmjz23rmrl7jt99e.streamlit.app/*

## Problem Statement

The Academic Risk Early Warning System predicts whether students are at risk of academic failure early in their academic journey. It uses attendance, test scores, and other academic metrics to make predictions. The goal is to provide early intervention and prevent academic failure.

## Key Results

The model is trained with a balanced dataset (about 50/50 SAFE vs AT RISK) and uses near-equal weighted academic factors to improve robustness.

## Tech Stack

- Python
- LightGBM (Gradient Boosting)
- Scikit-learn (Metrics & utilities)
- Pandas (Data Handling)
- Streamlit (Live UI demo)
- Joblib (Model deployment)
- imbalanced-learn (SMOTE)
- SHAP (Explainability)

## Live Demo

To run the live demo, follow these steps:

1. Install the project's dependencies:
```bash
pip install -r requirements.txt
```

2. Generate or refresh the dataset:
```bash
python generate_data.py
```

3. Train the model:
```bash
python model.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

You can expect to see real-time predictions, risk probability, and the top risk factors highlighted.

## Dataset

The dataset is synthetic and consists of 500 students (editable). It includes:
- internal_test1, internal_test2
- assignments_avg
- participation
- prev_sem_gpa
- attendance_pct
- engineered features: test_avg, risk_score

Labels are generated from a balanced, noisy risk score so SAFE/AT RISK are about 50/50.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
