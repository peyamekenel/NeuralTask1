import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

# Load and prepare the data
data = pd.read_csv('attachments/diabetes.csv')
features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[features]
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the scaler and other models
scaler = joblib.load('app/models/scaler.pkl')
models = {
    'svm': joblib.load('app/models/svm_model.pkl'),
    'logistic_regression': joblib.load('app/models/logistic_regression_model.pkl')
}

# Scale the features
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a new Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
models['random_forest'] = rf_model

def plot_confusion_matrix(model, X, y, model_name, save_path):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{model_name} Confusion Matrix:")
    print(cm)
    print(f"Number of incorrect predictions: {np.sum(y != y_pred)}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

# Generate and save confusion matrices for each model using test set
for model_name, model in models.items():
    plot_confusion_matrix(model, X_test_scaled, y_test,
                         model_name.replace('_', ' ').title(),
                         f'static_site/confusion_matrix_{model_name}.png')

# Save the properly trained Random Forest model
joblib.dump(rf_model, 'app/models/random_forest_model.pkl')
