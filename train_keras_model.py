import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Load and prepare the data
data = pd.read_csv('attachments/diabetes.csv')
features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[features]
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = joblib.load('app/models/scaler.pkl')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train Keras model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

# Save the Keras model
os.makedirs('app/models', exist_ok=True)
model.save('app/models/keras_model.keras')

# Generate predictions and confusion matrix using test set
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
cm = confusion_matrix(y_test, y_pred)

print("\nKeras Model Confusion Matrix:")
print(cm)
print(f"Number of incorrect predictions: {np.sum(y_test.values != y_pred)}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Keras')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('static_site/confusion_matrix_keras.png')
plt.close()
