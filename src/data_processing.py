import pandas as pd
import numpy as np
import os
from datetime import datetime

# Sub-task: Set up the environment
# Purpose: Create the output directory for processed data to ensure files are saved correctly.
def setup_environment():
    """Create the output directory for processed data."""
    os.makedirs('data/processed', exist_ok=True)
    print("Setup complete. Processed data will be saved to data/processed/.")

# Sub-task: Load the dataset
# Purpose: Read the raw transaction data and ensure it's ready for processing.
def load_data(file_path):
    """Load the dataset and display basic info."""
    df = pd.read_csv(file_path)
    print("\nLoaded dataset with shape:", df.shape)
    print("First 5 rows:")
    print(df.head())
    return df

# Sub-task: Handle missing values
# Purpose: Identify and fill missing values to ensure data quality for feature engineering.
def handle_missing_values(df):
    """Handle missing values based on EDA insights."""
    print("\nMissing Values Before Imputation:")
    print(df.isnull().sum())
    
    # Fill missing values: numerical with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    print("\nMissing Values After Imputation:")
    print(df.isnull().sum())
    return df

# Sub-task: Handle outliers
# Purpose: Cap outliers in numerical features like Amount to reduce their impact on the model.
def handle_outliers(df, numerical_cols):
    """Cap outliers in numerical features using IQR method."""
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"\nCapped outliers in {col} between {lower_bound} and {upper_bound}")
    return df

# Sub-task: Create RFM features
# Purpose: Calculate Recency, Frequency, and Monetary metrics per customer to capture their transaction behavior.
def create_rfm_features(df):
    """Create RFM features (Recency, Frequency, Monetary) at CustomerId level."""
    # Convert TransactionStartTime to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Define reference date (e.g., latest transaction date + 1 day)
    reference_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    # Calculate RFM
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (reference_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': 'sum'  # Monetary (sum of debits and credits)
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()
    
    print("\nRFM Features (first 5 rows):")
    print(rfm.head())
    return rfm

# Sub-task: Create additional features
# Purpose: Add features like transaction type, category counts, and average transaction amounts to enhance model inputs.
def create_additional_features(df):
    """Create additional features like transaction type and category counts."""
    # Transaction type (debit: Amount > 0, credit: Amount < 0)
    df['TransactionType'] = df['Amount'].apply(lambda x: 'Debit' if x > 0 else 'Credit')
    
    # Number of transactions per ProductCategory per CustomerId
    category_counts = df.groupby(['CustomerId', 'ProductCategory'])['TransactionId'].count().unstack(fill_value=0)
    category_counts.columns = [f'Count_{col}' for col in category_counts.columns]
    
    # Average Amount per CustomerId
    avg_amount = df.groupby('CustomerId')['Amount'].mean().reset_index(name='AvgTransactionAmount')
    
    print("\nTransaction Type Distribution:")
    print(df['TransactionType'].value_counts())
    print("\nCategory Counts (first 5 rows):")
    print(category_counts.head())
    print("\nAverage Transaction Amount (first 5 rows):")
    print(avg_amount.head())
    
    return df, category_counts, avg_amount

# Sub-task: Merge features
# Purpose: Combine RFM and additional features into a single dataset for modeling.
def merge_features(df, rfm, category_counts, avg_amount):
    """Merge RFM and additional features into a single dataset."""
    # Merge RFM with category counts and average amount
    processed_df = rfm.merge(category_counts, on='CustomerId', how='left')
    processed_df = processed_df.merge(avg_amount, on='CustomerId', how='left')
    
    print("\nMerged Features (first 5 rows):")
    print(processed_df.head())
    return processed_df

# Sub-task: Save processed dataset
# Purpose: Save the feature-engineered dataset to a CSV file for use in modeling.
def save_processed_data(df, output_path):
    """Save the processed dataset to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed dataset to {output_path}")

def main():
    """Main function to run feature engineering for Bati Bank's credit risk model."""
    print("Starting Feature Engineering for Bati Bank's Credit Risk Model...")
    
    # Run all sub-tasks
    setup_environment()
    df = load_data('data/raw/data.csv')
    
    # Handle missing values and outliers
    df = handle_missing_values(df)
    df = handle_outliers(df, ['Amount', 'Value'])
    
    # Create features
    rfm = create_rfm_features(df)
    df, category_counts, avg_amount = create_additional_features(df)
    processed_df = merge_features(df, rfm, category_counts, avg_amount)
    
    # Save processed dataset
    save_processed_data(processed_df, 'data/processed/processed_transactions.csv')

if __name__ == "__main__":
    main()