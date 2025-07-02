from pydantic import BaseModel

class InputData(BaseModel):
    """
    Pydantic model for input data validation.
    """
    features: list
