import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_data():
    
    #Load data from UCI ML Repository
    
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
    
    #Perform Exploratory Data Analysis
     
    print("\n--- Performing EDA ---")   

    # Combine features and target for analysis
    df = pd.concat([X, y], axis=1)
    
    print("\n1. Basic Dataset Information:")
    print("Dataset Shape:", df.shape)
    print("\nFeature Names:", list(X.columns))
    print("\nSample of the dataset:")
    print(df.head())
    
    print("\n2. Data Types and Missing Values:")
    print(df.info())
    
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    print("\n3. Missing Values Analysis:")
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.show()
    
    print("\n4. Categorical Variables Analysis:")
    for column in df.columns:
        print(f"\nDistribution of {column}:")
        value_counts = df[column].value_counts()
        print(value_counts)
        print(f"\nPercentage distribution of {column}:")
        print(value_counts / len(df) * 100)

        
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        value_counts.plot(kind='bar')
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

         # Add duplicate check
        print("\nDuplicate Rows:")
        duplicates = df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        
        # Add mode calculation for categorical variables
        print("\nMode for each feature:")
        print(df.mode().iloc[0])
    
    print("\n5. Target Variable Analysis:")
    print("Distribution of target variable (edible vs poisonous):")
    print(y.value_counts())
    print("\nPercentage distribution:")
    print(y.value_counts() / len(y) * 100)
    
    
    # Relationship between features and target
    print("\n6. Feature-Target Relationships:")
    for column in X.columns:
        contingency = pd.crosstab(X[column], y.iloc[:, 0])
        print(f"\nContingency table for {column}:")
        print(contingency)
        
        # Visualize relationship
        plt.figure(figsize=(10, 6))
        contingency.plot(kind='bar', stacked=True)
        plt.title(f'Relationship between {column} and Target')
        plt.xticks(rotation=45)
        plt.legend(title='Target')
        plt.tight_layout()
        plt.show()

def preprocess_data(X, y):
    
    #Preprocess the data
     
    print("\n7. Data Preprocessing:")
    
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
    
    print("\nEDA and Preprocessing completed!")


if __name__ == "__main__":
    main()
