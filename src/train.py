import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Sub-task: Set up the environment
# Purpose: Create directories for models and results, and initialize MLflow tracking.
def setup_environment():
    """Create directories and set up MLflow tracking."""
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), '../results'), exist_ok=True)
    mlflow.set_tracking_uri("file:" + os.path.abspath(os.path.join(os.path.dirname(__file__), '../mlruns')))
    mlflow.set_experiment("Credit_Risk_Model")
    print("Setup complete. MLflow tracking set to", mlflow.get_tracking_uri(), ", results to results/.")

# Sub-task: Load the processed dataset
# Purpose: Read the processed dataset with is_high_risk for model training.
def load_data(file_path):
    """Load the processed dataset and display basic info."""
    df = pd.read_csv(file_path)
    print("\nLoaded processed dataset with shape:", df.shape)
    print("First 5 rows:")
    print(df.head())
    return df

# Sub-task: Prepare data for modeling
# Purpose: Split data into features and target, and create training and test sets.
def prepare_data(df, target_col='is_high_risk'):
    """Split data into features (X) and target (y), and create train-test split."""
    X = df.drop(columns=[target_col, 'CustomerId'])  # Drop non-feature columns
    y = df[target_col]
    
    # Convert integer columns to float64 to handle potential missing values
    for col in X.columns:
        if X[col].dtype == 'int64':
            X[col] = X[col].astype('float64')
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print("\nTraining set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

# Sub-task: Train and tune model
# Purpose: Train a model with hyperparameter tuning using Grid Search and log to MLflow.
def train_and_tune_model(X_train, y_train, X_test, y_test, model_name, model, param_grid):
    """Train and tune a model, log results to MLflow."""
    with mlflow.start_run(run_name=f"{model_name}_Run"):
        # Perform Grid Search
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_pred)
        }
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        
        # Log model with input example for signature
        input_example = X_train.head(1)
        mlflow.sklearn.log_model(sk_model=best_model, artifact_path=f"{model_name}_model", input_example=input_example)
        
        print(f"\n{model_name} Best Parameters:", grid_search.best_params_)
        print(f"{model_name} Metrics:", metrics)
        return best_model, metrics

# Sub-task: Save evaluation metrics
# Purpose: Save model evaluation metrics to a CSV file for comparison.
def save_metrics(metrics_dict, output_path):
    """Save evaluation metrics to a CSV file."""
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(output_path, index=False)
    print(f"\nSaved evaluation metrics to {output_path}")

# Sub-task: Register the best model
# Purpose: Identify the best model based on F1-Score and register it in MLflow.
def register_best_model(metrics_dict, models_dict):
    """Register the model with the highest F1-Score in MLflow."""
    best_model_name = max(metrics_dict, key=lambda x: metrics_dict[x]['F1-Score'])
    best_model = models_dict[best_model_name]
    
    with mlflow.start_run(run_name=f"Register_{best_model_name}"):
        # Use correct feature names for input_example
        feature_names = ['Recency', 'Frequency', 'Monetary', 'Count_airtime', 'Count_data_bundles', 
                         'Count_financial_services', 'Count_movies', 'Count_other', 'Count_ticket', 
                         'Count_transport', 'Count_tv', 'Count_utility_bill', 'AvgTransactionAmount']
        input_example = pd.DataFrame([[0] * 13], columns=feature_names)
        mlflow.sklearn.log_model(sk_model=best_model, artifact_path=f"best_model_{best_model_name}", input_example=input_example)
        
        # Save the best model locally to models/
        joblib.dump(best_model, f"models/{best_model_name}.pkl")
        print(f"Saved best model to models/{best_model_name}.pkl")
        
        mlflow.set_tag("best_model", best_model_name)
        print(f"\nRegistered best model: {best_model_name} in MLflow Model Registry")
    return best_model_name

def main():
    """Main function to train and evaluate models for Bati Bank's credit risk model."""
    print("Starting Model Training and Tracking for Bati Bank's Credit Risk Model...")
    
    # Run all sub-tasks
    setup_environment()
    df = load_data('data/processed/processed_transactions_with_target.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Define models and parameter grids
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }
    
    # Train, tune, and evaluate models
    metrics_dict = {}
    models_dict = {}
    for model_name in models:
        best_model, metrics = train_and_tune_model(X_train, y_train, X_test, y_test, model_name, models[model_name], param_grids[model_name])
        metrics_dict[model_name] = metrics
        models_dict[model_name] = best_model
    
    # Save metrics
    save_metrics(metrics_dict, 'results/evaluation_metrics.csv')
    
    # Register the best model
    register_best_model(metrics_dict, models_dict)

if __name__ == "__main__":
    main()