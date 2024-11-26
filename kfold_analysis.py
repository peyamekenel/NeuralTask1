import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def preprocess_data(selected_features=None):
    df = pd.read_csv('attachments/diabetes.csv')

    # Handle zero values in medical measurements
    cols_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_process] = df[cols_to_process].replace(0, np.nan)
    for col in cols_to_process:
        df[col] = df[col].fillna(df[col].median())

    # Select features
    if selected_features:
        X = df[selected_features]
    else:
        X = df.drop('Outcome', axis=1)

    y = df['Outcome']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def perform_kfold_analysis(X, y, n_splits=10):
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize logistic regression
    lr = LogisticRegression(random_state=42)

    # Perform cross-validation for different metrics
    accuracy_scores = cross_val_score(lr, X, y, cv=kf, scoring='accuracy')
    precision_scores = cross_val_score(lr, X, y, cv=kf, scoring='precision')
    recall_scores = cross_val_score(lr, X, y, cv=kf, scoring='recall')
    f1_scores = cross_val_score(lr, X, y, cv=kf, scoring='f1')
    roc_auc_scores = cross_val_score(lr, X, y, cv=kf, scoring='roc_auc')

    # Create results dictionary
    results = {
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-score': f1_scores,
        'ROC AUC': roc_auc_scores
    }

    return results

def plot_kfold_results(all_features_results, selected_features_results):
    # Prepare data for plotting
    metrics = list(all_features_results.keys())

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot boxplots
    positions = np.arange(len(metrics)) * 3
    box1 = plt.boxplot([all_features_results[m] for m in metrics],
                      positions=positions - 0.5,
                      patch_artist=True)
    box2 = plt.boxplot([selected_features_results[m] for m in metrics],
                      positions=positions + 0.5,
                      patch_artist=True)

    # Customize boxplots
    for box in box1['boxes']:
        box.set(facecolor='lightblue')
    for box in box2['boxes']:
        box.set(facecolor='lightgreen')

    # Customize plot
    plt.xticks(positions, metrics, rotation=45)
    plt.ylabel('Score')
    plt.title('K-fold Cross-validation Results: All Features vs Selected Features')
    plt.legend([box1["boxes"][0], box2["boxes"][0]],
              ['All Features', 'Selected Features'],
              loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot
    plt.savefig('kfold_comparison.png')
    plt.close()

def print_detailed_results(results, model_name):
    print(f"\nDetailed results for {model_name}:")
    print("-" * 50)
    for metric, scores in results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric}:")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Std:  {std_score:.4f}")
        print(f"  Min:  {np.min(scores):.4f}")
        print(f"  Max:  {np.max(scores):.4f}")
        print()

if __name__ == "__main__":
    # Load data with all features
    X_all, y = preprocess_data()

    # Load data with selected features
    selected_features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X_selected, _ = preprocess_data(selected_features)

    # Perform K-fold analysis
    print("Performing 10-fold cross-validation...")
    all_features_results = perform_kfold_analysis(X_all, y)
    selected_features_results = perform_kfold_analysis(X_selected, y)

    # Print detailed results
    print_detailed_results(all_features_results, "Logistic Regression with All Features")
    print_detailed_results(selected_features_results, "Logistic Regression with Selected Features")

    # Plot results
    plot_kfold_results(all_features_results, selected_features_results)

    print("\nAnalysis complete. Results visualization saved as 'kfold_comparison.png'")
