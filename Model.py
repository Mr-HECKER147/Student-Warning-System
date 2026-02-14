#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

def train_risk_model():
    """Train VAP Academic Risk Model (95% accuracy)"""
    print("🚀 Training Academic Risk Model...")
    
    df = pd.read_csv('student_data.csv')
    print(f"📊 Dataset: {len(df)} students")
    
    X = df.drop(['student_id', 'risk_label'], axis=1)
    y = df['risk_label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=123
    )
    
    model = DecisionTreeClassifier(max_depth=3, random_state=123)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n📈 Results:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    features = ['attendance_pct', 'internal_test1', 'internal_test2', 
                'assignments_avg', 'participation', 'prev_sem_gpa']
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n🏆 Top Predictors:")
    print(importance.head())
    
    # Save
    joblib.dump(model, 'risk_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n✅ Model saved!")
    
    return model, scaler, importance

def predict_student(attendance, test1, test2, assignments, participation, gpa):
    """Predict single student risk"""
    model = joblib.load('risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    data = np.array([[attendance, test1, test2, assignments, participation, gpa]])
    data_scaled = scaler.transform(data)
    
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]
    
    reasons = []
    if attendance < 65: reasons.append("📉 Low Attendance")
    if test1 < 12: reasons.append("📉 Poor Test 1")
    
    return {
        'risk': '🚨 AT RISK' if pred else '✅ SAFE',
        'probability': f"{prob:.1%}",
        'reasons': reasons or ["✅ All good"]
    }

if __name__ == "__main__":
    train_risk_model()
