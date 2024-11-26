import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(selected_features):
    df = pd.read_csv('attachments/diabetes.csv')
    cols_to_process = [col for col in ['Glucose', 'BMI'] if col in selected_features]
    df[cols_to_process] = df[cols_to_process].replace(0, np.nan)
    for col in cols_to_process:
        df[col] = df[col].fillna(df[col].median())

    X = df[selected_features]
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def create_keras_model(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model_name, y_true, y_pred, y_pred_proba=None):
    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        results['auc'] = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {results["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} (Optimized)')
        plt.legend()
        plt.savefig(f'optimized_roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

    return results

if __name__ == "__main__":
    # Load selected features
    with open('selected_features.txt', 'r') as f:
        selected_features = f.read().splitlines()

    print("Using selected features:", selected_features)

    # Prepare data
    X_train, X_test, y_train, y_test = preprocess_data(selected_features)

    # Initialize and train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    results = []

    # Train and evaluate traditional ML models
    for name, model in models.items():
        print(f"\nTraining optimized {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        model_results = evaluate_model(name, y_test, y_pred, y_pred_proba)
        results.append(model_results)

    # Train and evaluate Keras model
    print("\nTraining optimized Keras model...")
    keras_model = create_keras_model(X_train.shape[1])
    history = keras_model.fit(X_train, y_train, epochs=100, batch_size=32,
                            validation_split=0.2, verbose=0)

    y_pred_proba = keras_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    keras_results = evaluate_model('Keras', y_test, y_pred, y_pred_proba)
    results.append(keras_results)

    # Create comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('model')
    print("\nOptimized Model Comparison (using selected features):")
    print(results_df.round(4))

    # Save results to CSV
    results_df.to_csv('optimized_model_comparison.csv')
