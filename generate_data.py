import pandas as pd
import numpy as np
np.random.seed(42)

n_students = 400
data = {
    'student_id': [f"S{str(i).zfill(3)}" for i in range(1, n_students + 1)],
    'attendance_pct': np.clip(np.random.normal(75, 15, n_students), 40, 100),
    'internal_test1': np.clip(np.random.normal(18, 5, n_students), 0, 25),
    'internal_test2': np.clip(np.random.normal(20, 4, n_students), 0, 25),
    'assignments_avg': np.clip(np.random.normal(14, 3, n_students), 0, 20),
    'participation': np.random.randint(1, 6, n_students),
    'prev_sem_gpa': np.clip(np.random.normal(7.5, 1.2, n_students), 4, 10)
}

df = pd.DataFrame(data)

# Generate realistic risk labels
df['risk_label'] = 0
df.loc[(df['attendance_pct'] < 65) | 
       (df['internal_test1'] < 12) | 
       (df['assignments_avg'] < 10), 'risk_label'] = 1

# Add some correlation for realism
df.loc[df['risk_label'] == 1, 'attendance_pct'] *= 0.85
df.loc[df['risk_label'] == 1, 'internal_test1'] *= 0.8
df.loc[df['risk_label'] == 1, 'assignments_avg'] *= 0.75

df.to_csv('student_data.csv', index=False)
print(f"Generated {len(df)} students. At Risk: {df['risk_label'].sum()} ({df['risk_label'].mean():.1%})")
