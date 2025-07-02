import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sub-task: Set up the environment
# Purpose: Create the output directory for predictions and set up MLflow tracking.
def setup_environment():
    """Create directory for predictions and set MLflow tracking URI."""
    os.makedirs('results', exist_ok=True)
    mlflow.set_tracking_uri("file:../mlruns")
    logging.info("Setup complete. MLflow tracking set to ../mlruns, predictions will be saved to results/.")

# Sub-task: Load the processed dataset
# Purpose: Read the input dataset for predictions (e.g., processed transactions).
def load_data(file_path):
    """Load the dataset and display basic info."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded dataset with shape: {df.shape}")
        logging.info("First 5 rows:\n%s", df.head())
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Sub-task: Prepare data for inference
# Purpose: Preprocess the input data to match the training data format (e.g., scale features).
def prepare_data(df, target_col='is_high_risk'):
    """Prepare data for model inference by scaling features."""
    try:
        # Define expected feature names
        feature_names = ['Recency', 'Frequency', 'Monetary', 'Count_airtime', 'Count_data_bundles', 
                         'Count_financial_services', 'Count_movies', 'Count_other', 'Count_ticket', 
                         'Count_transport', 'Count_tv', 'Count_utility_bill', 'AvgTransactionAmount']
        
        # Drop non-feature columns
        X = df[feature_names]  # Select only expected features
        
        # Convert integer columns to float64
        for col in X.columns:
            if X[col].dtype == 'int64':
                X[col] = X[col].astype('float64')
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        logging.info(f"Prepared data for inference with shape: {X_scaled.shape}")
        return X_scaled, df[['CustomerId']]
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

# Sub-task: Load the trained model
# Purpose: Load the best registered model from MLflow for predictions.
def load_model(model_name='best_model_GradientBoosting', run_id='<run_id>'):
    """Load the registered model from MLflow."""
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")
        logging.info(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

# Sub-task: Make predictions
# Purpose: Use the loaded model to predict is_high_risk for the input data.
def make_predictions(model, X, customer_ids):
    """Make predictions and return a DataFrame with CustomerId and predictions."""
    try:
        predictions = model.predict(X)
        result_df = pd.DataFrame({
            'CustomerId': customer_ids['CustomerId'],
            'is_high_risk_pred': predictions
        })
        logging.info("Predictions completed. First 5 predictions:\n%s", result_df.head())
        return result_df
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise

# Sub-task: Save predictions
# Purpose: Save the predictions to a CSV file for further analysis.
def save_predictions(predictions_df, output_path):
    """Save predictions to a CSV file."""
    try:
        predictions_df.to_csv(output_path, index=False)
        logging.info(f"Saved predictions to {output_path}")
    except Exception as e:
        logging.error(f"Error saving predictions: {e}")
        raise

def main():
    """Main function to perform model inference for Bati Bank's credit risk model."""
    logging.info("Starting Model Inference for Bati Bank's Credit Risk Model...")
    
    # Run all sub-tasks
    setup_environment()
    
    # Load data (use processed dataset or new data)
    input_file = 'data/processed/processed_transactions_with_target.csv'
    df = load_data(input_file)
    
    # Prepare data for inference
    X_scaled, customer_ids = prepare_data(df)
    
    # Load the best model from MLflow
    with mlflow.start_run(run_name="Inference_Run"):
        model = load_model(model_name='best_model_GradientBoosting', run_id='0')
        
        # Make predictions
        predictions_df = make_predictions(model, X_scaled, customer_ids)
        
        # Save predictions
        save_predictions(predictions_df, 'results/predictions.csv')

if __name__ == "__main__":
    main()