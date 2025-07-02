#  Credit Risk Model – Bati Bank (Buy Now, Pay Later)

Welcome to the **Credit Risk Model** project built for **Bati Bank’s Buy-Now-Pay-Later (BNPL)** service!  
This project harnesses real-world **eCommerce behavioral data** to:

-  Predict customer credit risk probabilities  
-  Assign credit scores  
-  Recommend suitable loan terms  

Powered by modern machine learning techniques, the model helps Bati Bank make **smart, fair, and regulatory-compliant lending decisions**.

---

## 📁 Project Structure

```bash
credit-risk-model-w5/
├── .github/workflows/ci.yml       # CI/CD pipeline config
├── data/                          # Raw and processed data (git-ignored)
│   ├── raw/                       # Raw datasets
│   └── processed/                 # Cleaned/engineered datasets
├── notebooks/
│   └── 1.0-eda.ipynb              # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering & preprocessing
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Model inference
│   └── api/
│       ├── main.py                # FastAPI app for predictions
│       └── pydantic_models.py     # API data schemas
├── tests/
│   └── test_data_processing.py    # Unit tests for pipeline
├── Dockerfile                     # Docker setup
├── docker-compose.yml             # Docker Compose environment
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files/folders to ignore in git
└── README.md                      # Project documentation
````

---

##  Business Understanding

### Why does the Basel II Accord emphasize interpretability?

Basel II is a global banking regulation that encourages **risk transparency and accountability**.
Our model aligns with this by being:

* **Transparent** – Stakeholders (regulators/customers) can understand the reasoning behind credit decisions.
* **Auditable** – Easy to evaluate, making compliance checks smoother.
* **Safe** – Clear documentation helps avoid missteps and reinforces best practices.

### Why use a proxy variable for default?

We lack a direct "default" label, so we engineered a **proxy target** using **Recency, Frequency, and Monetary (RFM)** patterns.

*  **Why it's useful**: Allows us to train a model despite missing true labels.
*  **Risks**: If the proxy poorly reflects actual defaults, the model may:

  * Approve risky borrowers (losses)
  * Reject good borrowers (missed revenue)

 **Mitigation**: Regular validation and updates to the proxy logic.

### Simple vs. Complex Models: What’s the Trade-off?

| Model Type                                  | Pros                                           | Cons                                           |
| ------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| **Simple (e.g. Logistic Regression + WoE)** | Transparent, regulator-friendly, interpretable | May miss complex patterns in data              |
| **Complex (e.g. Gradient Boosting)**        | High predictive power, captures subtle signals | Harder to explain, risk of regulatory pushback |

 We balance **clarity** with **accuracy**, prioritizing trust and compliance while maintaining strong performance.

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd credit-risk-model-w5
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI service (via Docker)**

   ```bash
   docker-compose up
   ```

---

## Usage Guide

| Task                    | Command/Instruction                                         |
| ----------------------- | ----------------------------------------------------------- |
|  Exploratory Analysis | Open `notebooks/1.0-eda.ipynb` in Jupyter                   |
|  Data Processing      | Run: `python src/data_processing.py`                        |
|  Train Model          | Run: `python src/train.py`                                  |
|  Make Predictions     | Run: `python src/predict.py` or call `/predict` via FastAPI |

---

##  Final Notes

This project was developed as part of the **10 Academy Week 5 Challenge**, using 95,000+ rows of anonymized customer behavior data. It demonstrates how **ethical AI + regulatory alignment** can empower fintech innovation.

