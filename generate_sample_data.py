"""
Generate sample customer churn data for testing
"""
import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

print("Generating sample customer churn data...")

# Generate customer IDs
customer_ids = [f'CUST{str(i).zfill(6)}' for i in range(1, n_samples + 1)]

# Generate features
data = {
    'customerID': customer_ids,
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    'tenure': np.random.randint(0, 73, n_samples),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.45, 0.15]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
    'PaymentMethod': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n_samples
    ),
    'MonthlyCharges': np.round(np.random.uniform(18.0, 120.0, n_samples), 2),
}

# Calculate TotalCharges based on tenure and MonthlyCharges
data['TotalCharges'] = [
    str(round(tenure * monthly, 2)) if tenure > 0 else ''
    for tenure, monthly in zip(data['tenure'], data['MonthlyCharges'])
]

# Generate Churn labels (target variable)
# Higher churn probability for:
# - Month-to-month contracts
# - High monthly charges
# - Low tenure
churn_probs = []
for i in range(n_samples):
    prob = 0.2  # Base probability
    if data['Contract'][i] == 'Month-to-month':
        prob += 0.3
    if data['tenure'][i] < 12:
        prob += 0.2
    if data['MonthlyCharges'][i] > 80:
        prob += 0.15
    churn_probs.append(min(prob, 0.9))

data['Churn'] = [np.random.choice(['Yes', 'No'], p=[prob, 1-prob]) for prob in churn_probs]

# Create DataFrame
df = pd.DataFrame(data)

print(f"Generated {len(df)} samples")
print(f"\nChurn distribution:")
print(df['Churn'].value_counts())

# Save to CSV
output_dir = 'data/raw'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'customer_churn_data.csv')
df.to_csv(output_path, index=False)
print(f"\nDataset saved to {output_path}")

print("\nFirst few rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
