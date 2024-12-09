from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def split_data(X, y):
    #Split the dataset into training and testing sets

    return train_test_split(X, y, test_size=0.2, random_state=42)

def hyperparameter_tuning(X_train, y_train):

    #Perform hyperparameter tuning for each model using GridSearchCV with cross-validation
    # Define models and their hyperparameters
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        },
        'NaiveBayes': {
            'model': GaussianNB(),
            'params': {}  # No hyperparameters for Naive Bayes
        }
    }
    
    best_models = {}  # To store the best model for each classifier
    for name, m in models.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(
            estimator=m['model'],
            param_grid=m['params'],
            cv=5,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_  # Save the best model
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}\n")
    
    return best_models
