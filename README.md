# Customer Churn Prediction Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://end-to-end-customer-churn-prediction-zytxfcgsgrmxs9zyvses2p.streamlit.app/#customer-churn-prediction-system)

### 1) Problem Statement

**The Business Context:** In the highly competitive telecommunications sector, customer retention is the single most critical driver of profitability. Industry statistics indicate that acquiring a new customer costs 5 to 25 times more than retaining an existing one. Furthermore, a mere 5% increase in customer retention can boost overall profits by 25% to 95%. [[Harvard Business Review](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers?utm_source)]

**The Pain Point:** "Telco X" is currently experiencing a high churn rate, where valuable customers are terminating their subscriptions in favor of competitors. This "silent leak" in revenue is not only increasing Customer Acquisition Costs (CAC) but also eroding Customer Lifetime Value (CLV).

**The Solution:** Traditional retention strategies (e.g., mass marketing emails) are inefficient because they target the wrong people. This project implements a precision-based Machine Learning solution to:

1.  **Predict** exactly which customers are at high risk of churning in the next month.
2.  **Identify** the specific root causes for each customer (e.g., "Fiber Optic user with no Tech Support").
3.  **Enable** the support team to take proactive, targeted action (e.g., offering a specific discount) before the customer leaves.

### 2) Data Collection

**Dataset Source:** [IBM Dataset (Telco customer churn)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data)

**The data consists of 33 columns and 7043 rows.**

### 3) 📊 Exploratory Data Analysis (EDA)

We conducted a thorough analysis to understand the data landscape:

- **Class Imbalance:** The dataset is imbalanced, with approximately **26%** of customers churning. This necessitated specific handling techniques during modeling.
- **Tenure:** New customers are significantly more likely to churn. The median tenure for churners is around 10 months, compared to over 30 months for retained customers.
- **Contract Type:** Customers with **Month-to-month** contracts have a much higher churn rate compared to those on one or two-year contracts.

### 4) 🤖 Model Development

We experimented with several models to find the best balance between identifying churners (Recall) and overall accuracy:

1.  **Logistic Regression (Baseline):** Good accuracy but struggled to catch enough churners.
2.  **Random Forest:** Improved performance but still biased toward the majority class.
3.  **XGBoost:** Tuned for high recall but resulted in too many false positives (low precision).
4.  **Neural Network (Final Model):**
    - **Architecture:** Multi-layer Perceptron with Dropout for regularization.
    - **Technique:** Used `class_weight` to penalize the model more for missing churners.
    - **Performance:**
      - **Accuracy:** ~76%
      - **Recall (Churn):** ~80% (Successfully identifies 8 out of 10 churners)

### 5) 🔑 Key Drivers of Churn

Based on our Feature Importance analysis (Permutation Importance), the top factors driving churn are:

- **Tenure Months:** Low tenure is the biggest risk factor.
- **Total Charges:** High total spend (often correlated with tenure) is a factor.
- **Contract Type:** Month-to-month contracts are highly risky.
- **Internet Service:** Fiber optic users churn more frequently (possibly due to price or service quality).
- **Payment Method:** Electronic check users show higher churn rates.

### 6) 🚀 How to Run

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

    This will launch the web interface where you can input customer details and get churn predictions.

3.  **Train the Model (Optional):**
    If you want to retrain the model from scratch:
    ```bash
    python train.py
    ```
    This script will:
    - Ingest and preprocess the data.
    - Train the Neural Network model.
    - Save the artifacts (`model.h5` and `preprocessor.pkl`) in the `artifacts/` folder.

### 7) 📂 Project Structure

```
.
├── artifacts
│   ├── model.h5
│   └── preprocessor.pkl
├── data
│   └── raw
│       └── Telco_customer_churn.xlsx
├── notebook
│   ├── 1.EDA_and_insights.ipynb
│   ├── 2.model_training.ipynb
│   └── churn_model.h5
├── src
│   ├── components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline
│   │   ├── prediction_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── requirements.txt
├── setup.py
├── train.py
└── README.md
```

### 8) 📊 Deployment

The app is deployed on Streamlit Cloud. You can access it [here](https://end-to-end-customer-churn-prediction-zytxfcgsgrmxs9zyvses2p.streamlit.app/#customer-churn-prediction-system).
