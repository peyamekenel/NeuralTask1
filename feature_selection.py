import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def preprocess_data():
    df = pd.read_csv('attachments/diabetes.csv')
    cols_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_process] = df[cols_to_process].replace(0, np.nan)
    for col in cols_to_process:
        df[col] = df[col].fillna(df[col].median())
    return df

# Get feature importance using Random Forest
def get_feature_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    return importance_df

# Select top features
def select_top_features(X, y, threshold=0.1):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(rf, threshold=threshold)
    selector.fit(X, y)

    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features

if __name__ == "__main__":
    # Load and preprocess data
    df = preprocess_data()

    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Get feature importance
    importance_df = get_feature_importance(X, y)
    print("\nFeature Importance:")
    print(importance_df)

    # Select top features
    selected_features = select_top_features(X, y)
    print("\nSelected Features:")
    print(selected_features)

    # Save selected features
    with open('selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))
