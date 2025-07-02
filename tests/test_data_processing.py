import pytest
from src.data_processing import create_pipeline

def test_pipeline_creation():
    """
    Test that the data processing pipeline is created successfully.
    """
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.steps) == 3  # Imputer, encoder, scaler
