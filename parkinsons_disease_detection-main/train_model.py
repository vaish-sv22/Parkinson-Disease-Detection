# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("D:/PD Project/parkinsons_data.csv")

# Features used for training
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
    'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
    'NHR', 'HNR'
]

X = df[features]
y = df['status']

print("Class distribution before SMOTE:\n", y.value_counts())

# -----------------------------
# Balance dataset using SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("\nClass distribution after SMOTE:\n", pd.Series(y_res).value_counts())

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Define base models
# -----------------------------
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
svm_clf = SVC(kernel='rbf', probability=True, random_state=42)

# -----------------------------
# Voting Classifier
# -----------------------------
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('svm', svm_clf)],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

print("\n=== Final Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save model + scaler
# -----------------------------
joblib.dump(voting_clf, "parkinsons_voting_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")