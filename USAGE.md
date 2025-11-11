# Usage Guide - Customer Churn Prediction System

This guide provides step-by-step instructions for using the Customer Churn Prediction system.

## Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Data Preparation](#data-preparation)
3. [Training the Model](#training-the-model)
4. [Making Predictions](#making-predictions)
5. [Web Application](#web-application)
6. [API Usage](#api-usage)
7. [Docker Deployment](#docker-deployment)

---

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Abhaykumar2021/End-to-End-Customer-Churn-Prediction.git
cd End-to-End-Customer-Churn-Prediction
```

2. **Create and activate virtual environment:**
```bash
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package in development mode:**
```bash
pip install -e .
```

---

## Data Preparation

### Generate Sample Data

If you don't have customer churn data, you can generate sample data:

```bash
python generate_sample_data.py
```

This creates a sample dataset at `data/raw/customer_churn_data.csv` with 1000 customer records.

### Data Format

Your CSV file should include the following columns:

**Customer Information:**
- `customerID`: Unique customer identifier (optional)
- `gender`: Male/Female
- `SeniorCitizen`: 0/1 (0=No, 1=Yes)
- `Partner`: Yes/No
- `Dependents`: Yes/No
- `tenure`: Number of months customer has been with the company

**Services:**
- `PhoneService`: Yes/No
- `MultipleLines`: Yes/No/No phone service
- `InternetService`: DSL/Fiber optic/No
- `OnlineSecurity`: Yes/No/No internet service
- `OnlineBackup`: Yes/No/No internet service
- `DeviceProtection`: Yes/No/No internet service
- `TechSupport`: Yes/No/No internet service
- `StreamingTV`: Yes/No/No internet service
- `StreamingMovies`: Yes/No/No internet service

**Account Information:**
- `Contract`: Month-to-month/One year/Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: Electronic check/Mailed check/Bank transfer/Credit card

**Charges:**
- `MonthlyCharges`: Monthly charge amount (float)
- `TotalCharges`: Total charge amount (string, converted to float)

**Target (for training only):**
- `Churn`: Yes/No

---

## Training the Model

### Basic Training

Train the model with your data:

```bash
python train.py --data data/raw/customer_churn_data.csv
```

### What Happens During Training

1. **Data Ingestion**: Data is loaded and split into train (80%) and test (20%) sets
2. **Data Validation**: Schema validation and drift detection
3. **Data Transformation**: Feature engineering and preprocessing
4. **Model Training**: Multiple models are trained and compared:
   - Logistic Regression
   - Random Forest
   - XGBoost
5. **Model Selection**: Best model is selected based on accuracy
6. **Model Saving**: Model and preprocessor are saved to `artifacts/`

### Training Output

```
artifacts/
└── TIMESTAMP/
    ├── data_ingestion/
    │   ├── feature_store/
    │   └── ingested/
    ├── data_validation/
    │   └── drift_report/
    ├── data_transformation/
    │   ├── transformed/
    │   └── transformed_object/
    └── model_trainer/
        └── trained_model/
            └── model.pkl
```

### Check Training Logs

Logs are saved in the `logs/` directory:

```bash
cat logs/*.log | grep "Best model"
```

---

## Making Predictions

### Command Line Predictions

1. **Prepare your test data** (CSV file without the Churn column)

2. **Run predictions:**
```bash
python predict.py --input test_data.csv --output predictions.csv
```

3. **View results:**
```bash
head predictions.csv
```

The output will include:
- All original columns
- `Predicted_Churn`: 0 (No) or 1 (Yes)
- `Churn_Probability`: Probability of churn (0-1)

### Example Prediction

```python
import pandas as pd
from churn_prediction.pipeline.prediction_pipeline import PredictionPipeline, CustomData

# Create sample customer data
customer = CustomData(
    gender="Female",
    SeniorCitizen=0,
    Partner="Yes",
    Dependents="No",
    tenure=12,
    PhoneService="Yes",
    MultipleLines="No",
    InternetService="Fiber optic",
    OnlineSecurity="No",
    OnlineBackup="Yes",
    DeviceProtection="No",
    TechSupport="No",
    StreamingTV="Yes",
    StreamingMovies="Yes",
    Contract="Month-to-month",
    PaperlessBilling="Yes",
    PaymentMethod="Electronic check",
    MonthlyCharges=70.35,
    TotalCharges="844.2"
)

# Convert to DataFrame
df = customer.get_data_as_dataframe()

# Make prediction
pipeline = PredictionPipeline()
predictions, probabilities = pipeline.predict(df)

print(f"Churn Prediction: {'Yes' if predictions[0] == 1 else 'No'}")
print(f"Churn Probability: {probabilities[0][1]:.2%}")
```

---

## Web Application

### Starting the Web App

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface

1. **Home Page**: Navigate to `http://localhost:5000`
   - View system overview
   - Click "Make a Prediction" button

2. **Prediction Page**: `http://localhost:5000/predict`
   - Fill in customer details form
   - Click "Predict Churn"
   - View prediction result and confidence

### Web Application Features

- ✅ User-friendly form interface
- ✅ Input validation
- ✅ Instant predictions
- ✅ Confidence scores
- ✅ Responsive design

---

## API Usage

### REST API Endpoint

**Endpoint:** `POST /api/predict`

**Request Format:**
```json
{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": "844.2"
}
```

**Response Format:**
```json
{
    "churn": true,
    "confidence": 0.73,
    "probabilities": {
        "no_churn": 0.27,
        "churn": 0.73
    }
}
```

### Example API Call (Python)

```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    # ... other fields
}

response = requests.post(url, json=data)
result = response.json()

print(f"Will churn: {result['churn']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Example API Call (cURL)

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": "844.2"
  }'
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t churn-prediction:latest .
```

### Run Docker Container

```bash
docker run -p 5000:5000 churn-prediction:latest
```

### Access the Application

Navigate to `http://localhost:5000` in your browser.

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./saved_models:/app/saved_models
    environment:
      - FLASK_ENV=production
```

Run with:
```bash
docker-compose up
```

---

## Troubleshooting

### Model Not Found Error

If you get "Model not found" error:
1. Ensure you've trained the model first: `python train.py --data your_data.csv`
2. Check that `saved_models/model.pkl` exists

### Import Errors

If you encounter import errors:
```bash
pip install -r requirements.txt --upgrade
```

### Data Format Issues

Ensure your CSV file:
- Has all required columns
- Uses correct value formats (Yes/No, 0/1, etc.)
- Has no missing customerID values if included

### Permission Issues

On Linux/Mac, if you encounter permission issues:
```bash
chmod +x train.py predict.py
```

---

## Project Structure Reference

```
End-to-End-Customer-Churn-Prediction/
├── churn_prediction/          # Main package
│   ├── components/            # Pipeline components
│   ├── pipeline/              # Training & prediction pipelines
│   ├── config/                # Configuration management
│   ├── entity/                # Data classes
│   ├── constants/             # Constants
│   ├── logger/                # Logging
│   ├── exception/             # Exception handling
│   └── utils/                 # Utility functions
├── config/                    # YAML configurations
├── data/                      # Data directory
├── templates/                 # HTML templates
├── static/                    # CSS, JS files
├── logs/                      # Log files
├── artifacts/                 # Training artifacts
├── saved_models/              # Production models
├── app.py                     # Flask application
├── train.py                   # Training script
├── predict.py                 # Prediction script
└── generate_sample_data.py    # Data generator
```

---

## Additional Resources

- **README.md**: Project overview and installation
- **Jupyter Notebook**: `notebooks/sample_data_generator.ipynb` for interactive data generation
- **Configuration Files**: 
  - `config/schema.yaml`: Data schema definition
  - `config/model.yaml`: Model hyperparameters
- **Logs**: Check `logs/` directory for detailed execution logs

---

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review the error messages for specific issues
3. Ensure all dependencies are correctly installed
4. Verify data format matches the expected schema

---

**Note**: This is a machine learning project. Model performance depends on the quality and quantity of training data. For production use, train with real customer data and validate the model performance thoroughly.
