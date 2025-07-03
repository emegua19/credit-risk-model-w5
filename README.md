# Credit Risk Model – Bati Bank (Buy Now, Pay Later)

A machine learning solution to assess credit risk, assign scores, and recommend loan amounts using eCommerce data — compliant with Basel II regulatory standards.

---

## Project Overview

This project was developed as part of the **10 Academy Week 5 Challenge** to provide Bati Bank with a BNPL (Buy Now, Pay Later) credit risk scoring system. It uses 95,663 real eCommerce transaction records to:

* Predict credit risk probability
* Assign customer credit scores
* Recommend loan limits and durations

The model pipeline ensures interpretability, compliance with the **Basel II Capital Accord**, and is built for production with FastAPI and Docker.

---

## Project Structure

```
credit-risk-model-w5/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb
├── outputs/
│   ├── logs/
│   └── plots/
│       └── eda_plots/
├── src/
│   ├── data_processing.py
│   ├── target_variable_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
├── models/
├── mlruns/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Setup

### Prerequisites

* Python 3.10+
* Docker
* Git

### Steps

```bash
# Clone the repository
git clone https://github.com/emegua19/credit-risk-model-w5.git
cd credit-risk-model-w5

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app
docker-compose up --build
```

---

## Usage Instructions

### API Endpoints

| Endpoint   | Method | Description                         |
| ---------- | ------ | ----------------------------------- |
| `/`        | GET    | Health check endpoint               |
| `/predict` | POST   | Predicts credit risk for a customer |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict?customer_id=Customer_123" \
-H "Content-Type: application/json" \
-d '{"Recency": 84, "Frequency": 1, "Monetary": 4325.0, ...}'
```

### CLI Commands

* Run EDA: `jupyter notebook notebooks/1.0-eda.ipynb`
* Data processing: `python src/data_processing.py`
* Proxy target: `python src/target_variable_engineering.py`
* Model training: `python src/train.py`
* Inference: `python src/predict.py`
* Tests: `pytest tests/`

---

## Business Understanding

### Basel II Compliance

**Pillar 1:** Requires at least 8% capital against RWAs; IRB models must be documented and validated.
**Pillar 2:** Supervisors assess model adequacy through stress testing and governance.
**Pillar 3:** Banks must disclose risks, models, and capital strategies for public transparency.

### Proxy Target: Why & How

* **Problem:** No "default" label in dataset.
* **Solution:** Created `is_high_risk` using KMeans clustering on RFM scores.
* **Validation:** Cross-referenced with `FraudResult` labels.
* **Risk Mitigation:** Documented methodology and used proxy diagnostics.

### Model Strategy

* **Simple Model:** Logistic Regression with WoE (compliant, explainable)
* **Complex Model:** Gradient Boosting (high performance, logged via MLflow)
* **Metrics:** ROC-AUC, Precision, Recall, Confusion Matrix

---

## Task Summary

### Task 1 – Business Understanding

* Reviewed Basel II Accord
* Defined project objectives and constraints
* Chose interpretable models as baseline

### Task 2 – EDA

* Used `DataExplorer` in Jupyter
* Generated distribution, correlation, boxplots
* Found: 40% negative Amounts, 0.2% FraudResult, no nulls, strong skew

### Task 3 – Feature Engineering

* Built `DataProcessor` class
* Added `IsNegativeAmount`, extracted time features
* Log-transformed skewed fields

### Task 4 – Target Engineering

* RFM clustering with `RFMClustering`
* Labeled least engaged group as `is_high_risk = 1`

### Task 5 – Modeling & Evaluation

* Used `train.py` for training Logistic & Gradient Boosting
* Logged via MLflow in `mlruns/`
* Applied SMOTE and class weights

### Task 6 – API Deployment

* Built FastAPI app in `main.py`
* Served predictions via `/predict` endpoint
* Dockerized with `docker-compose.yml`

---

## Exploratory Data Insights

| Key Insight           | Observation                                       |
| --------------------- | ------------------------------------------------- |
| Class Imbalance       | Only 0.2% of FraudResult > 0                      |
| Refund Detection      | 40% of transactions are refunds (negative Amount) |
| No Missing Data       | All fields complete                               |
| Right Skewed Features | `Amount`, `Value` require transformation          |
| Feature Redundancy    | Amount & Value highly correlated                  |
| Drop Constant Feature | `CountryCode` = 256 for all records               |

---

## Pipeline Overview

```text
┌────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌──────────────┐
│  Raw Data  │ → │ Data Preprocessing │ → │ RFM Proxy Clustering│ → │ Model Training     │ → │ API Inference       │ → │ Deployment   │
└────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘   └──────────────┘
```

---

## Testing & CI/CD

* Unit tested with `pytest`
* Automated CI/CD via `.github/workflows/ci.yml`

---

## References

* [Basel II – Investopedia](https://www.investopedia.com/terms/b/baselii.asp)
* [Proxy Variables – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/how-to-use-proxy-variables-in-a-regression-model/)
* [SHAP for Explainability](https://github.com/slundberg/shap)
* [Credit Risk Modeling – TDS](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)

---

## Author

**Yitbarek Geletaw**
