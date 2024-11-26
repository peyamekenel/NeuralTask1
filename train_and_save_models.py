import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('app/models', exist_ok=True)

# Load and prepare the data
data = pd.read_csv('attachments/diabetes.csv')
features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[features]
y = data['Outcome']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
models = {
    'random_forest': RandomForestClassifier(random_state=42),
    'svm': SVC(probability=True, random_state=42),
    'logistic_regression': LogisticRegression(random_state=42)
}

# Train and save each model
for name, model in models.items():
    model.fit(X_scaled, y)
    joblib.dump(model, f'app/models/{name}_model.pkl')

# Save the scaler
joblib.dump(scaler, 'app/models/scaler.pkl')
