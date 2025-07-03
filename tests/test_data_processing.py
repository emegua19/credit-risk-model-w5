import os
import sys
import pandas as pd

# Calculate paths relative to this test file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# Import after path modification (with flake8 exception)
from src.data_processing import handle_missing_values  # noqa: E402


def test_handle_missing_values_numerical():
    """Test numerical missing values are filled with median."""
    data = pd.DataFrame({
        "Amount": [1000, 2000, None, 4000],
        "ProductCategory": [
            "airtime",
            "financial_services",
            "airtime",
            None,
        ],
    })
    result = handle_missing_values(data.copy())

    assert result["Amount"].isna().sum() == 0, (
        "Numerical column should have no missing values"
    )
    assert result["Amount"][2] == 2000, (
        "Missing numerical value should be filled with median (2000)"
    )


def test_handle_missing_values_categorical():
    """Test categorical missing values are filled with mode."""
    data = pd.DataFrame({
        "Amount": [1000, 2000, None, 4000],
        "ProductCategory": [
            "airtime",
            "financial_services",
            "airtime",
            None,
        ],
    })
    result = handle_missing_values(data.copy())

    assert result["ProductCategory"].isna().sum() == 0, (
        "Categorical column should have no missing values"
    )
    assert result["ProductCategory"][3] == "airtime", (
        "Missing categorical value should be filled with mode (airtime)"
    )
