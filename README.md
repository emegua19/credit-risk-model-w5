# Credit Risk Model

This project implements a credit scoring model for Bati Bank's buy-now-pay-later service, using transactional data from an eCommerce platform. The model predicts the probability of default, assigns credit scores, and recommends loan terms.

## Project Structure
```
credit-risk-model-w5/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── data/                      # Raw and processed data (ignored by git)
│   ├── raw/                   # Raw datasets
│   └── processed/             # Processed datasets
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Feature engineering
│   ├── train.py               # Model training
│   ├── predict.py             # Model inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # Project documentation
```

## Credit Scoring Business Understanding
*To be completed as part of Task 1.*

## Setup Instructions
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the FastAPI service: `docker-compose up`

## Usage
- Run EDA: Open `notebooks/1.0-eda.ipynb` in Jupyter.
- Process data: Execute `src/data_processing.py`.
- Train model: Run `src/train.py`.
- Make predictions: Use `src/predict.py` or the FastAPI endpoint `/predict`.
