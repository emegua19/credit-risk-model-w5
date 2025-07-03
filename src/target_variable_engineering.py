import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Sub-task: Set up the environment
# Purpose: Create the output directory for the processed dataset with the
# new target variable.
def setup_environment():
    """Create the output directory for processed data."""
    os.makedirs("../data/processed", exist_ok=True)
    print("Setup complete. Processed data will be saved to data/processed/.")


# Sub-task: Load the raw dataset
# Purpose: Read the raw transaction data to calculate RFM metrics accurately.
def load_data(file_path):
    """Load the raw dataset and display basic info."""
    df = pd.read_csv(file_path)
    print("\nLoaded raw dataset with shape:", df.shape)
    print("First 5 rows:")
    print(df.head())
    return df


# Sub-task: Calculate RFM metrics
# Purpose: Compute Recency, Frequency, and Monetary values per CustomerId
# to capture transaction behavior.
def calculate_rfm_metrics(df):
    """Calculate RFM metrics (Recency, Frequency, Monetary)."""
    # Convert TransactionStartTime to datetime
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Define snapshot date (latest transaction date + 1 day)
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    # Calculate RFM
    rfm = (
        df.groupby("CustomerId")
        .agg(
            {
                "TransactionStartTime": lambda x: (
                    snapshot_date - x.max()
                ).days,  # Recency
                "TransactionId": "count",  # Frequency
                "Amount": "sum",  # Monetary (sum of debits and credits)
            }
        )
        .rename(
            columns={
                "TransactionStartTime": "Recency",
                "TransactionId": "Frequency",
                "Amount": "Monetary",
            }
        )
        .reset_index()
    )

    print("\nRFM Metrics (first 5 rows):")
    print(rfm.head())
    return rfm, snapshot_date


# Sub-task: Pre-process RFM features
# Purpose: Scale RFM features to ensure fair clustering.
def preprocess_rfm_features(rfm):
    """Scale RFM features using StandardScaler."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    rfm_scaled_df = pd.DataFrame(
        rfm_scaled, columns=["Recency", "Frequency", "Monetary"]
    )
    rfm_scaled_df["CustomerId"] = rfm["CustomerId"]
    print("\nScaled RFM Features (first 5 rows):")
    print(rfm_scaled_df.head())
    return rfm_scaled_df, scaler


# Sub-task: Cluster customers
# Purpose: Use K-Means to segment customers into 3 groups.
def cluster_customers(rfm_scaled_df, n_clusters=3, random_state=42):
    """Apply K-Means clustering to segment customers."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_scaled_df["Cluster"] = kmeans.fit_predict(
        rfm_scaled_df[["Recency", "Frequency", "Monetary"]]
    )
    print("\nCluster Distribution:")
    print(rfm_scaled_df["Cluster"].value_counts())
    return rfm_scaled_df, kmeans


# Sub-task: Identify high-risk cluster
# Purpose: Analyze clusters to find the high-risk group.
def identify_high_risk_cluster(rfm, rfm_scaled_df):
    """Identify the high-risk cluster and create is_high_risk column."""
    # Merge original RFM values with cluster labels
    rfm_with_clusters = rfm.merge(
        rfm_scaled_df[["CustomerId", "Cluster"]], on="CustomerId"
    )

    # Calculate mean RFM values per cluster
    cluster_summary = (
        rfm_with_clusters.groupby("Cluster")
        .agg({"Recency": "mean", "Frequency": "mean", "Monetary": "mean"})
        .reset_index()
    )
    print("\nCluster Summary (mean RFM values):")
    print(cluster_summary)

    # Identify high-risk cluster: low Frequency and low Monetary
    high_risk_cluster = cluster_summary[
        (cluster_summary["Frequency"] == cluster_summary["Frequency"].min())
        | (cluster_summary["Monetary"] == cluster_summary["Monetary"].min())
    ]["Cluster"].iloc[0]
    print(f"\nHigh-risk cluster identified: Cluster {high_risk_cluster}")

    # Assign is_high_risk (1 for high-risk cluster, 0 otherwise)
    rfm_with_clusters["is_high_risk"] = (
        rfm_with_clusters["Cluster"] == high_risk_cluster
    ).astype(int)
    print("\nis_high_risk Distribution:")
    print(rfm_with_clusters["is_high_risk"].value_counts())

    return rfm_with_clusters[["CustomerId", "is_high_risk"]]


# Sub-task: Merge target variable with processed dataset
# Purpose: Integrate is_high_risk into the processed dataset.
def merge_target_variable(processed_df, target_df):
    """Merge is_high_risk column into the processed dataset."""
    merged_df = processed_df.merge(target_df, on="CustomerId", how="left")
    print("\nMerged Dataset with is_high_risk (first 5 rows):")
    print(merged_df.head())
    return merged_df


# Sub-task: Save processed dataset
# Purpose: Save the updated dataset with the is_high_risk column.
def save_processed_data(df, output_path):
    """Save the processed dataset to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed dataset to {output_path}")


def main():
    """Main function to engineer the proxy target variable."""
    print("Starting Proxy Target Variable Engineering...")

    # Run all sub-tasks
    setup_environment()

    # Load raw and processed datasets
    raw_df = load_data("data/raw/data.csv")
    processed_df = load_data("data/processed/processed_transactions.csv")

    # Calculate RFM and cluster
    rfm, snapshot_date = calculate_rfm_metrics(raw_df)
    rfm_scaled_df, scaler = preprocess_rfm_features(rfm)
    rfm_scaled_df, kmeans = cluster_customers(rfm_scaled_df)

    # Identify high-risk cluster and create is_high_risk
    target_df = identify_high_risk_cluster(rfm, rfm_scaled_df)

    # Merge and save
    merged_df = merge_target_variable(processed_df, target_df)
    save_processed_data(
        merged_df, "data/processed/processed_transactions_with_target.csv"
    )


if __name__ == "__main__":
    main()
