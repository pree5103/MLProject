from mushrooms_processing import load_data, perform_eda, preprocess_data
from mushrooms_models import split_data, hyperparameter_tuning
from mushrooms_eval import evaluate_models

def main():
    # Load data
    X, y, metadata, variables = load_data()
    
    # Perform EDA
    perform_eda(X, y) 
    
    # Preprocess data
    X_processed, y_processed = preprocess_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)
    
    # Perform hyperparameter tuning
    best_models = hyperparameter_tuning(X_train, y_train)
    
    # Evaluate models (pass feature names)
    feature_names = X_train.columns  # Use the column names from your processed dataset
    metrics_df = evaluate_models(best_models, X_test, y_test, feature_names=feature_names)
    
    # Print metrics summary
    print(metrics_df)


if __name__ == "__main__":
    main()
