import pandas as pd
import numpy as np
import os

np.random.seed(42)

n_students = 2000
data = {
    'student_id': [f"S{str(i).zfill(4)}" for i in range(1, n_students + 1)],
    'attendance_pct':  np.clip(np.random.normal(75, 15, n_students), 40, 100),
    'internal_test1':  np.clip(np.random.normal(18,  5, n_students),  0,  25),
    'internal_test2':  np.clip(np.random.normal(20,  4, n_students),  0,  25),
    'assignments_avg': np.clip(np.random.normal(14,  3, n_students),  0,  20),
    'participation':   np.random.randint(1, 6, n_students),
    'prev_sem_gpa':    np.clip(np.random.normal(7.5, 1.2, n_students), 4, 10),
}

df = pd.DataFrame(data)

# Engineered features
df['test_avg']   = (df['internal_test1'] + df['internal_test2']) / 2
df['risk_score'] = (
    df['attendance_pct']  * 0.35 +
    df['test_avg']        * 0.30 +
    df['prev_sem_gpa']    * 0.20 +
    df['assignments_avg'] * 0.15
)

# Risk labels based on OR conditions
df['risk_label'] = 0
df.loc[
    (df['attendance_pct']  < 65) |
    (df['internal_test1']  < 12) |
    (df['internal_test2']  < 12) |
    (df['assignments_avg'] < 10) |
    (df['participation']   < 3)  |
    (df['prev_sem_gpa']    < 6.0),
    'risk_label'
] = 1

# Add realistic correlation for at-risk students
idx = df['risk_label'] == 1
df.loc[idx, 'attendance_pct']  *= 0.85
df.loc[idx, 'internal_test1']  *= 0.80
df.loc[idx, 'assignments_avg'] *= 0.75
df.loc[idx, 'participation']    = np.random.randint(1, 3, idx.sum())
df.loc[idx, 'prev_sem_gpa']    *= 0.90

# Recalculate engineered features after correlation adjustment
df['test_avg']   = (df['internal_test1'] + df['internal_test2']) / 2
df['risk_score'] = (
    df['attendance_pct']  * 0.35 +
    df['test_avg']        * 0.30 +
    df['prev_sem_gpa']    * 0.20 +
    df['assignments_avg'] * 0.15
)

os.makedirs('data', exist_ok=True)
df.to_csv('data/student_data.csv', index=False)
print(f"Generated {len(df)} students.")
print(f"At Risk : {df['risk_label'].sum()} ({df['risk_label'].mean():.1%})")
print(f"Safe    : {(df['risk_label']==0).sum()} ({(df['risk_label']==0).mean():.1%})")
