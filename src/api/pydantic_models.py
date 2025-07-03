
from pydantic import BaseModel, validator
from typing import List

class CustomerData(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    Count_airtime: float
    Count_data_bundles: float
    Count_financial_services: float
    Count_movies: float
    Count_other: float
    Count_ticket: float
    Count_transport: float
    Count_tv: float
    Count_utility_bill: float
    AvgTransactionAmount: float

    @validator('Recency', 'Frequency', 'Monetary', 'Count_airtime', 'Count_data_bundles', 
                'Count_financial_services', 'Count_movies', 'Count_other', 'Count_ticket', 
                'Count_transport', 'Count_tv', 'Count_utility_bill', 'AvgTransactionAmount')
    def positive_values(cls, v):
        if v < 0:
            raise ValueError(f"{v} must be non-negative")
        return v

class PredictionResponse(BaseModel):
    CustomerId: str
    is_high_risk_prob: float

    class Config:
        json_schema_extra = {
            "example": {
                "CustomerId": "CustomerId_1",
                "is_high_risk_prob": 0.95
            }
        }