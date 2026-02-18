import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

def train_risk_model():
    print("🚀 Training Model with Percentage Contributions...")
    
    df = pd.read_csv('student_data.csv')
    
    # 1. Features in your specific priority order
    features = ['internal_test1', 'internal_test2', 'attendance_pct', 
                'assignments_avg', 'participation', 'prev_sem_gpa']
    
    X = df[features]
    y = df['risk_label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    # 2. Assigning weights to match your hierarchy (1 is highest, 6 is lowest)
    # Higher number = more influence on the result.
    custom_weights = np.array([[-5.0, -4.0, -3.0, -2.5, -2.0, -1.5]])
    model.coef_ = custom_weights
    model.intercept_ = np.array([5.0]) 

    # 3. CALCULATE PERCENTAGE CONTRIBUTION
    # We take the absolute value of weights and divide by the total sum
    total_weight = np.sum(np.abs(custom_weights))
    percentages = (np.abs(custom_weights) / total_weight) * 100

    # 4. Create the importance DataFrame
    importance = pd.DataFrame({
        'Factor': features,
        'Weight': custom_weights[0],
        'Contribution (%)': percentages[0]
    }).sort_values('Contribution (%)', ascending=False)

    print("\n🏆 Factor Contribution Breakdown:")
    # Formatting to show % sign
    importance_display = importance.copy()
    importance_display['Contribution (%)'] = importance_display['Contribution (%)'].map('{:,.2f}%'.format)
    print(importance_display[['Factor', 'Contribution (%)']])

    # Save
    joblib.dump(model, 'risk_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n✅ Model and Scaler saved!")

def predict_student(attendance, test1, test2, assignments, participation, gpa):
    model = joblib.load('risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    input_data = np.array([[test1, test2, attendance, assignments, participation, gpa]])
    input_scaled = scaler.transform(input_data)
    
    risk_prob = model.predict_proba(input_scaled)[0][1]
    risk_label = "AT RISK" if risk_prob > 0.5 else "SAFE"
    
    reasons = []
    if attendance < 65:
        reasons.append("Low Attendance")
    if test1 < 12:
        reasons.append("Poor Internal Test 1")
    if test2 < 12:
        reasons.append("Poor Internal Test 2")
    if assignments < 10:
        reasons.append("Poor Assignments")
    if participation < 3:
        reasons.append("Low Participation")
    if gpa < 6:
        reasons.append("Low Previous GPA")
    
    return {
        'risk': risk_label,
        'probability': f"{risk_prob:.2%}",
        'reasons': reasons
    }

if __name__ == "__main__":
    train_risk_model()