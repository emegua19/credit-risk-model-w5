import os
import sys
import pandas as pd

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

from src.data_processing import handle_missing_values


def test_handle_missing_values_numerical():
    """Test numerical missing values are filled with median."""
    # Create sample DataFrame with missing values
    data = pd.DataFrame(
        {
            "Amount": [1000, 2000, None, 4000],
            "ProductCategory": [
                "airtime",
                "financial_services",
                "airtime",
                None
            ],
        }
    )
    result = handle_missing_values(data.copy())

    # Expected median for Amount (1000, 2000, 4000) = 2000
    assert result["Amount"].isna().sum() == 0, (
        "Numerical column should have no missing values"
    )
    assert result["Amount"][2] == 2000, (
        "Missing numerical value should be filled with median (2000)"
    )


def test_handle_missing_values_categorical():
    """Test categorical missing values are filled with mode."""
    # Create sample DataFrame with missing values
    data = pd.DataFrame(
        {
            "Amount": [1000, 2000, None, 4000],
            "ProductCategory": [
                "airtime",
                "financial_services",
                "airtime",
                None
            ],
        }
    )
    result = handle_missing_values(data.copy())

    # Expected mode for ProductCategory = airtime
    assert result["ProductCategory"].isna().sum() == 0, (
        "Categorical column should have no missing values"
    )
    assert result["ProductCategory"][3] == "airtime", (
        "Missing categorical value should be filled with mode (airtime)"
    )
