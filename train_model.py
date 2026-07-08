"""
train_model.py
----------------
Yeh script aapke notebook (student_prediction_model.ipynb) ki training pipeline
ka clean version hai. Isay aap sirf ek dafa chalayenge taake
`student_prediction_model.pkl` file ban jaye — wohi file Streamlit app use karegi.

USAGE:
    1. Is file ko usi folder mein rakho jahan "student_performance_dataset.csv" hai
    2. Terminal mein run karo:  python train_model.py
    3. "student_prediction_model.pkl" ban jayegi
"""

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------
df = pd.read_csv("student_performance_dataset.csv")

# ------------------------------------------------------------------
# 2. Drop ID column
# ------------------------------------------------------------------
df.drop("Student_ID", axis=1, inplace=True)

# ------------------------------------------------------------------
# 3. Encode categorical columns (same order/logic as your notebook)
# ------------------------------------------------------------------
le = LabelEncoder()
categorical_columns = df.select_dtypes(include="object").columns
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# ------------------------------------------------------------------
# 4. Features / Target
# ------------------------------------------------------------------
X = df.drop(["Pass_Fail", "Final_Exam_Score"], axis=1)
y = df["Pass_Fail"]

# ------------------------------------------------------------------
# 5. Train / Test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 6. GridSearchCV -> Best Random Forest (same params as notebook)
# ------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
best_pred = best_rf.predict(X_test)

print("Best Parameters:", grid_rf.best_params_)
print("Test Accuracy  :", accuracy_score(y_test, best_pred))

# ------------------------------------------------------------------
# 7. Save Model
# ------------------------------------------------------------------
joblib.dump(best_rf, "student_prediction_model.pkl")
print("\n✅ Model Saved Successfully as student_prediction_model.pkl")
