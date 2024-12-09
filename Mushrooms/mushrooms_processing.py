import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def load_data():
 
    # Fetch dataset
    mushroom = fetch_ucirepo(id=73)
    
    # Get data and metadata
    X = mushroom.data.features
    y = mushroom.data.targets
    
    print("Dataset Metadata:")
    print(mushroom.metadata)
    print("\nVariable Information:")
    print(mushroom.variables)
    
    return X, y, mushroom.metadata, mushroom.variables


def perform_eda(X, y):
 
    print("\n--- Performing EDA ---")

    # Combine features and target for analysis
    df = pd.concat([X, y], axis=1)
    
    # Basic Dataset Information
    print("\n1. Basic Dataset Information:")
    print("Dataset Shape:", df.shape)
    print("\nFeature Names:", list(X.columns))
    print("\nSample of the dataset:")
    print(df.head())
    
    # Data Types and Missing Values
    print("\n2. Data Types and Missing Values:")
    print(df.info())
    
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    print("\nMissing Values Analysis:")
    print(missing_values[missing_values > 0])
    
    # Visualize missing values if any
    if missing_values.sum() > 0:
        plt.figure(figsize=(10, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.show()
    
    # Target Variable Distribution
    print("\n3. Target Variable Analysis:")
    print("Target Distribution (edible vs poisonous):")
    print(y.value_counts())
    print("\nPercentage distribution:")
    print((y.value_counts() / len(y)) * 100)
    
    # Feature Distributions 
    print("\n4. Feature Distributions:")
    for column in X.columns:
        print(f"\nDistribution of {column}:")
        print(X[column].value_counts())
        print(f"\nPercentage distribution of {column}:")
        print((X[column].value_counts() / len(X)) * 100)

    # Check for Duplicate Rows
    print("\n5. Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")


def preprocess_data(X, y):
     
    print("\n--- Preprocessing Data ---")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Encode categorical variables
    le = LabelEncoder()
    X_encoded = X_imputed.copy()
    for column in X_encoded.columns:
        X_encoded[column] = le.fit_transform(X_encoded[column])
    
    y_encoded = le.fit_transform(y.iloc[:, 0])
    
    print("Shape after preprocessing:", X_encoded.shape)
    print("\nSample of preprocessed features:")
    print(X_encoded.head())
    print("\nSample of preprocessed target:")
    print(y_encoded[:5])
    
    return X_encoded, y_encoded


def main():
    # Load data
    X, y, metadata, variables = load_data()
    
    # Perform EDA
    perform_eda(X, y)
    
    # Preprocess data
    X_processed, y_processed = preprocess_data(X, y)
    
    print("\nEDA and Preprocessing completed.")
  


if __name__ == "__main__":
    main()
