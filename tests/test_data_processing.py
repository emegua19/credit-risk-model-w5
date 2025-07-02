import pandas as pd
import pytest
from src.data_processing import handle_missing_values

# Test 1: Check if numerical missing values are filled with median
def test_handle_missing_values_numerical():
    """Test that numerical missing values are filled with the median."""
    # Create a sample DataFrame with missing values
    data = pd.DataFrame({
        'Amount': [1000, 2000, None, 4000],
        'ProductCategory': ['airtime', 'financial_services', 'airtime', None]
    })
    result = handle_missing_values(data.copy())
    
    # Expected median for Amount (1000, 2000, 4000) = 2000
    assert result['Amount'].isna().sum() == 0, "Numerical column should have no missing values"
    assert result['Amount'][2] == 2000, "Missing numerical value should be filled with median (2000)"

# Test 2: Check if categorical missing values are filled with mode
def test_handle_missing_values_categorical():
    """Test that categorical missing values are filled with the mode."""
    # Create a sample DataFrame with missing values
    data = pd.DataFrame({
        'Amount': [1000, 2000, None, 4000],
        'ProductCategory': ['airtime', 'financial_services', 'airtime', None]
    })
    result = handle_missing_values(data.copy())
    
    # Expected mode for ProductCategory (airtime, financial_services, airtime) = airtime
    assert result['ProductCategory'].isna().sum() == 0, "Categorical column should have no missing values"
    assert result['ProductCategory'][3] == 'airtime', "Missing categorical value should be filled with mode (airtime)"