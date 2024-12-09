from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_models(models, X_test, y_test, feature_names):
    
    # DataFrame to store metrics
    metrics_summary = []

    for name, model in models.items():
        # Predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Classification metrics
        print(f"Evaluation for {name}:")
        report = classification_report(y_test, y_pred, output_dict=True)  # Get metrics as dictionary
        print(classification_report(y_test, y_pred))  # Print full report
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Extract key metrics
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1_score = report["weighted avg"]["f1-score"]
        metrics_summary.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        })

        # ROC AUC and ROC curve (if applicable)
        if y_prob is not None:
            auc = roc_auc_score(y_test, y_prob)
            print(f"ROC AUC: {auc}")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {name}')
            plt.legend(loc='best')
            plt.show()

        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            print(f"\nFeature Importance for {name}:")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            print(importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.bar(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title(f'Feature Importance for {name}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    # Display summary metrics for all models
    print("\nSummary of Metrics for All Models:")
    metrics_df = pd.DataFrame(metrics_summary)
    print(metrics_df)

    return metrics_df  # Return metrics as a DataFrame for further use
