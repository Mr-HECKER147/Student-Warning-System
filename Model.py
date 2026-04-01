import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not installed. Run: pip install imbalanced-learn")
    print("   Training without SMOTE oversampling.\n")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Run: pip install shap")
    print("   Risk reasons will fall back to threshold-based logic.\n")


# ── Feature order (must be identical everywhere) ─────────────────────────────
FEATURES = [
    'internal_test1',
    'internal_test2',
    'attendance_pct',
    'assignments_avg',
    'participation',
    'prev_sem_gpa',
    'test_avg',
    'risk_score',
]

# ── Fallback thresholds if SHAP unavailable ───────────────────────────────────
THRESHOLDS = {
    'internal_test1':  (12,  "Poor Internal Test 1 score"),
    'internal_test2':  (12,  "Poor Internal Test 2 score"),
    'attendance_pct':  (65,  "Low Attendance"),
    'assignments_avg': (10,  "Poor Assignment scores"),
    'participation':   (3,   "Low Class Participation"),
    'prev_sem_gpa':    (6.0, "Low Previous Semester GPA"),
    'test_avg':        (12,  "Low average test score"),
    'risk_score':      (50,  "Overall risk score too low"),
}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(test1, test2, attendance, assignments, participation, gpa):
    checks = [
        (test1,         0,  25,  "Internal Test 1 must be 0–25"),
        (test2,         0,  25,  "Internal Test 2 must be 0–25"),
        (attendance,    0, 100,  "Attendance must be 0–100"),
        (assignments,   0,  20,  "Assignments must be 0–20"),
        (participation, 1,   5,  "Participation must be 1–5"),
        (gpa,           0,  10,  "GPA must be 0–10"),
    ]
    for value, low, high, msg in checks:
        if not (low <= value <= high):
            raise ValueError(f"Invalid input — {msg}. Got: {value}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_risk_model():
    print("🚀 Training Risk Model with LightGBM...\n")

    df = pd.read_csv('data/student_data.csv')

    missing = [f for f in FEATURES + ['risk_label'] if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    X = df[FEATURES]
    y = df['risk_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Apply SMOTE to balance classes on training set
    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
        print(f"✅ SMOTE applied. Training samples: {len(y_train)}")

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )

    y_pred   = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')

    print(f"✅ Test Accuracy      : {accuracy:.2%}")
    print(f"✅ Cross-Val Accuracy : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}\n")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["SAFE", "AT RISK"]))
    print("🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature':        FEATURES,
        'Importance':     importances,
        'Contribution %': (importances / importances.sum() * 100).round(2)
    }).sort_values('Contribution %', ascending=False)
    print("\n🏆 Feature Importances:")
    print(imp_df[['Feature', 'Contribution %']].to_string(index=False))

    os.makedirs('model', exist_ok=True)
    joblib.dump(model,  'model/risk_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')

    model_info = {
        'framework':         'LightGBM',
        'accuracy':          round(accuracy, 4),
        'cv_accuracy_mean':  round(cv_scores.mean(), 4),
        'cv_accuracy_std':   round(cv_scores.std(), 4),
        'trained_on':        datetime.now().strftime('%Y-%m-%d %H:%M'),
        'training_samples':  len(X_train_scaled),
        'smote_applied':     SMOTE_AVAILABLE,
        'feature_importances': dict(zip(FEATURES, importances.round(4).tolist()))
    }
    with open('model/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("\n✅ Saved: model/risk_model.pkl, model/scaler.pkl, model/model_info.json")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_student(test1, test2, attendance, assignments, participation, gpa):
    validate_inputs(test1, test2, attendance, assignments, participation, gpa)

    model  = joblib.load('model/risk_model.pkl')
    scaler = joblib.load('model/scaler.pkl')

    # Build engineered features to match training
    test_avg   = (test1 + test2) / 2
    risk_score = (
        attendance    * 0.35 +
        test_avg      * 0.30 +
        gpa           * 0.20 +
        assignments   * 0.15
    )

    input_data   = np.array([[test1, test2, attendance, assignments,
                               participation, gpa, test_avg, risk_score]])
    input_scaled = scaler.transform(input_data)

    risk_prob  = model.predict_proba(input_scaled)[0][1]
    risk_label = "AT RISK" if risk_prob > 0.5 else "SAFE"

    reasons     = []
    shap_values = None

    if SHAP_AVAILABLE:
        explainer   = shap.TreeExplainer(model)
        shap_output = explainer.shap_values(input_scaled)
<<<<<<< HEAD

        # shap_values for class 1 (AT RISK)
        if isinstance(shap_output, list):
            # Old format: list of arrays for binary classification
            sv = shap_output[1][0]
        else:
            # New format: 3D array (n_samples, n_features, n_classes)
            sv = shap_output[0, :, 1]

=======
        sv = shap_output[1][0] if isinstance(shap_output, list) else shap_output[0]
>>>>>>> 50cb35f25276feb1eb012def2637b2c841719316
        shap_values = dict(zip(FEATURES, sv.tolist()))
        sorted_shap = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
        reasons = [
            f"{feat.replace('_', ' ').title()} is a key risk factor"
            for feat, val in sorted_shap[:3]
            if val > 0
        ]
    else:
        values = {
            'internal_test1':  test1,
            'internal_test2':  test2,
            'attendance_pct':  attendance,
            'assignments_avg': assignments,
            'participation':   participation,
            'prev_sem_gpa':    gpa,
            'test_avg':        test_avg,
            'risk_score':      risk_score,
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
        'shap_values': shap_values,
    }


if __name__ == "__main__":
    train_risk_model()
    predict_student(18, 20, 75, 14, 3, 7.5)