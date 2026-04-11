"""
Student Performance Prediction - Simplified Model Training
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("=" * 80)
print(" STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
print("=" * 80)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('data/student_data_final.csv')
X = df.drop(columns=['passed', 'G3', 'performance_level', 'needs_improvement'])
y = df['passed']
print(f"✓ Loaded {len(df)} students with {len(X.columns)} features")

# Encode categorical variables
print("\n[2/5] Encoding categorical variables...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}")

joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("✓ Encoders saved")

# Split data
print("\n[3/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Training: {len(X_train)} samples, Testing: {len(X_test)} samples")

# Scale features
print("\n[4/5] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Scaler saved")

# Train models
print("\n[5/5] Training models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}
best_model = None
best_auc = 0

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
    
    print(f"    Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_model = name

# Save best model
best_model_obj = models[best_model]
joblib.dump(best_model_obj, 'models/best_model.pkl')
print(f"\n✓ Best model ({best_model}) saved")

# Save metrics
metrics = {
    'best_model': best_model,
    'accuracy': results[best_model]['accuracy'],
    'precision': results[best_model]['precision'],
    'recall': results[best_model]['recall'],
    'f1_score': results[best_model]['f1'],
    'auc_roc': results[best_model]['auc']
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🏆 Best Model: {best_model}")
print(f"📊 Test Accuracy: {metrics['accuracy']:.4f}")
print(f"📊 AUC-ROC: {metrics['auc_roc']:.4f}")