import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Run: pip install shap")
    print("   Risk reasons will fall back to threshold-based logic.\n")


# ── Feature order (must be identical everywhere) ─────────────────────────────
FEATURES = [
    'internal_test1',   # /25
    'internal_test2',   # /25
    'attendance_pct',   # 0–100
    'assignments_avg',  # /20
    'participation',    # 1–5
    'prev_sem_gpa'      # 0–10
]

# ── Thresholds used ONLY as fallback if SHAP is unavailable ──────────────────
THRESHOLDS = {
    'internal_test1':  (12,  "Poor Internal Test 1 score"),
    'internal_test2':  (12,  "Poor Internal Test 2 score"),
    'attendance_pct':  (65,  "Low Attendance"),
    'assignments_avg': (10,  "Poor Assignment scores"),
    'participation':   (3,   "Low Class Participation"),
    'prev_sem_gpa':    (6.0, "Low Previous Semester GPA"),
}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(test1, test2, attendance, assignments, participation, gpa):
    """Raises ValueError if any input is outside its valid range."""
    checks = [
        (test1,         0, 25,  "Internal Test 1 must be between 0 and 25"),
        (test2,         0, 25,  "Internal Test 2 must be between 0 and 25"),
        (attendance,    0, 100, "Attendance must be between 0 and 100"),
        (assignments,   0, 20,  "Assignments must be between 0 and 20"),
        (participation, 1, 5,   "Participation must be between 1 and 5"),
        (gpa,           0, 10,  "GPA must be between 0 and 10"),
    ]
    for value, low, high, msg in checks:
        if not (low <= value <= high):
            raise ValueError(f"Invalid input — {msg}. Got: {value}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_risk_model():
    print("🚀 Training Risk Model...\n")

    df = pd.read_csv('data/student_data.csv')

    # Verify all required columns exist
    missing = [f for f in FEATURES + ['risk_label'] if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    X = df[FEATURES]
    y = df['risk_label']

    # ── Train / Test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)   # fit only on train, transform test

    # ── Train Random Forest ───────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'   # handles imbalanced at-risk vs safe counts
    )
    model.fit(X_train_scaled, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred    = model.predict(X_test_scaled)
    accuracy  = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)

    print(f"✅ Test Accuracy      : {accuracy:.2%}")
    print(f"✅ Cross-Val Accuracy : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}\n")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["SAFE", "AT RISK"]))
    print("🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Feature importances (from the model, not manually assigned) ───────────
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature':        FEATURES,
        'Importance':     importances,
        'Contribution %': (importances / importances.sum() * 100).round(2)
    }).sort_values('Contribution %', ascending=False)

    print("\n🏆 Real Feature Importances:")
    print(importance_df[['Feature', 'Contribution %']].to_string(index=False))

    # ── Save model artifacts ──────────────────────────────────────────────────
    joblib.dump(model,  'model/risk_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')

    model_info = {
        'accuracy':          round(accuracy, 4),
        'cv_accuracy_mean':  round(cv_scores.mean(), 4),
        'cv_accuracy_std':   round(cv_scores.std(), 4),
        'trained_on':        datetime.now().strftime('%Y-%m-%d %H:%M'),
        'training_samples':  len(X_train),
        'feature_importances': dict(
            zip(FEATURES, importances.round(4).tolist())
        )
    }

    with open('model/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("\n✅ Saved: risk_model.pkl, scaler.pkl, model_info.json")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_student(test1, test2, attendance, assignments, participation, gpa):
    """
    Returns a dict with:
      - risk:        "AT RISK" or "SAFE"
      - probability: string like "73.21%"
      - reasons:     list of human-readable contributing factors
      - shap_values: raw SHAP values per feature (for charting in app.py)
    """
    # 1. Validate
    validate_inputs(test1, test2, attendance, assignments, participation, gpa)

    # 2. Load artifacts
    model  = joblib.load('model/risk_model.pkl')
    scaler = joblib.load('model/scaler.pkl')

    # 3. Build input — order MUST match FEATURES list
    input_data   = np.array([[test1, test2, attendance, assignments, participation, gpa]])
    input_scaled = scaler.transform(input_data)

    # 4. Predict
    risk_prob  = model.predict_proba(input_scaled)[0][1]
    risk_label = "AT RISK" if risk_prob > 0.5 else "SAFE"

    # 5. Explain with SHAP (preferred) or threshold fallback
    reasons     = []
    shap_values = None

    if SHAP_AVAILABLE:
        explainer   = shap.TreeExplainer(model)
        shap_output = explainer.shap_values(input_scaled)

        # shap_values for class 1 (AT RISK)
        sv = shap_output[1][0] if isinstance(shap_output, list) else shap_output[0]
        shap_values = dict(zip(FEATURES, sv.tolist()))

        # Top 3 features pushing toward AT RISK (positive SHAP = pushes toward risk)
        sorted_shap = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
        reasons = [
            f"{feat.replace('_', ' ').title()} is a key risk factor"
            for feat, val in sorted_shap[:3]
            if val > 0
        ]
    else:
        # Fallback: simple threshold checks
        values = {
            'internal_test1':  test1,
            'internal_test2':  test2,
            'attendance_pct':  attendance,
            'assignments_avg': assignments,
            'participation':   participation,
            'prev_sem_gpa':    gpa,
        }
        for feat, (threshold, message) in THRESHOLDS.items():
            if values[feat] < threshold:
                reasons.append(message)

    if not reasons:
        reasons = ["All indicators are within acceptable range"]

    return {
        'risk':        risk_label,
        'probability': f"{risk_prob:.2%}",
        'reasons':     reasons,
        'shap_values': shap_values   # used by app.py to draw the bar chart
    }


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_risk_model()
