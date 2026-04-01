import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

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
    'risk_score':      (60,  "Overall risk score too low"),
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

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    # Apply SMOTE to balance classes on training set
    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applied. Training samples: {len(y_train)}")

    model = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_val, (val_probs >= t).astype(int)) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1_scores))])

    test_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, test_probs)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')

    print(f"Test Accuracy      : {accuracy:.2%}")
    print(f"Test F1 Score      : {f1:.2%}")
    print(f"Test ROC-AUC       : {auc:.3f}")
    print(f"Cross-Val F1       : {cv_scores.mean():.2%} +/- {cv_scores.std():.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["SAFE", "AT RISK"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature contribution distribution (more stable than raw gain)
    contributions = None
    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_output = explainer.shap_values(X_test)
        if isinstance(shap_output, list):
            sv = shap_output[1]
        else:
            if len(shap_output.shape) == 3:
                sv = shap_output[:, :, 1]
            else:
                sv = shap_output
        contrib_vals = np.mean(np.abs(sv), axis=0)
    else:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='f1')
        contrib_vals = perm.importances_mean

    if contrib_vals.sum() == 0:
        contrib_pct = np.zeros_like(contrib_vals)
    else:
        contrib_pct = contrib_vals / contrib_vals.sum() * 100

    contributions = dict(zip(FEATURES, contrib_pct.round(2).tolist()))
    contrib_df = pd.DataFrame({
        'Feature': FEATURES,
        'Contribution %': [contributions[f] for f in FEATURES],
    }).sort_values('Contribution %', ascending=False)
    print("\nFeature Contribution Distribution:")
    print(contrib_df.to_string(index=False))


    os.makedirs('model', exist_ok=True)
    joblib.dump(model,  'model/risk_model.pkl')

    model_info = {
        'framework':         'LightGBM',
        'accuracy':          round(accuracy, 4),
        'f1':                round(f1, 4),
        'roc_auc':           round(auc, 4),
        'cv_f1_mean':        round(cv_scores.mean(), 4),
        'cv_f1_std':         round(cv_scores.std(), 4),
        'best_threshold':    round(best_threshold, 4),
        'trained_on':        datetime.now().strftime('%Y-%m-%d %H:%M'),
        'training_samples':  len(X_train),
        'smote_applied':     SMOTE_AVAILABLE,
        'feature_contributions': contributions
    }
    with open('model/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("\nSaved: model/risk_model.pkl, model/model_info.json")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_student(test1, test2, attendance, assignments, participation, gpa):
    validate_inputs(test1, test2, attendance, assignments, participation, gpa)

    model = joblib.load('model/risk_model.pkl')
    threshold = 0.5
    try:
        with open('model/model_info.json', 'r') as f:
            model_info = json.load(f)
            threshold = float(model_info.get('best_threshold', 0.5))
    except Exception:
        pass

    # Build engineered features to match training
    test_avg   = (test1 + test2) / 2
    risk_score = (
        0.18 * (test1 / 25.0) +
        0.18 * (test2 / 25.0) +
        0.18 * (assignments / 20.0) +
        0.18 * (gpa / 10.0) +
        0.14 * (attendance / 100.0) +
        0.14 * (participation / 5.0)
    ) * 100

    input_data = pd.DataFrame([[
        test1, test2, attendance, assignments,
        participation, gpa, test_avg, risk_score
    ]], columns=FEATURES)

    risk_prob  = model.predict_proba(input_data)[0][1]
    risk_label = "AT RISK" if risk_prob >= threshold else "SAFE"

    reasons     = []
    shap_values = None

    if SHAP_AVAILABLE:
        explainer   = shap.TreeExplainer(model)
        shap_output = explainer.shap_values(input_data)


        # shap_values for class 1 (AT RISK)
        if isinstance(shap_output, list):
            # Old format: list of arrays for binary classification
            sv = shap_output[1][0]
        else:
            # Newer formats: either (n_samples, n_features, n_classes)
            # or (n_samples, n_features)
            if len(shap_output.shape) == 3:
                sv = shap_output[0, :, 1]
            else:
                sv = shap_output[0]
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
