# End-to-End Customer Churn Prediction

A complete end-to-end machine learning project for predicting customer churn using various ML algorithms.

## Project Overview

This project implements a complete machine learning pipeline for predicting customer churn. It includes data ingestion, validation, transformation, model training, and deployment with a web interface.

## Features

- **Data Ingestion**: Automated data loading and splitting into train/test sets
- **Data Validation**: Schema validation and data drift detection
- **Data Transformation**: Feature engineering and preprocessing pipeline
- **Model Training**: Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive model evaluation metrics
- **Web Application**: Flask-based web interface for predictions
- **REST API**: API endpoint for programmatic access
- **Logging**: Comprehensive logging system
- **Exception Handling**: Custom exception handling

## Project Structure

```
End-to-End-Customer-Churn-Prediction/
├── churn_prediction/           # Main package
│   ├── components/             # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/               # Training and prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── config/                 # Configuration management
│   ├── entity/                 # Configuration entities
│   ├── constants/              # Project constants
│   ├── logger/                 # Logging module
│   ├── exception/              # Exception handling
│   └── utils/                  # Utility functions
├── config/                     # Configuration files
│   ├── schema.yaml             # Data schema
│   └── model.yaml              # Model configurations
├── data/                       # Data directory
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── artifacts/                  # Training artifacts
├── notebooks/                  # Jupyter notebooks
├── templates/                  # HTML templates
├── static/                     # Static files (CSS, JS)
├── logs/                       # Log files
├── app.py                      # Flask application
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── requirements.txt            # Python dependencies
└── setup.py                    # Package setup

```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abhaykumar2021/End-to-End-Customer-Churn-Prediction.git
cd End-to-End-Customer-Churn-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

## Usage

### Training the Model

To train the model with your own data:

```bash
python train.py --data path/to/your/data.csv
```

The training pipeline will:
1. Ingest and split the data
2. Validate the data schema and check for drift
3. Transform and preprocess the data
4. Train multiple models with hyperparameter tuning
5. Select and save the best model

### Making Predictions

#### Using the Command Line

```bash
python predict.py --input path/to/test_data.csv --output predictions.csv
```

#### Using the Web Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Use the web interface to:
   - View the home page
   - Navigate to the prediction page
   - Enter customer details
   - Get instant churn predictions

#### Using the REST API

Send a POST request to `/api/predict` with customer data:

```python
import requests
import json

url = "http://localhost:5000/api/predict"
data = {
    "gender": "Male",
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

response = requests.post(url, json=data)
print(response.json())
```

## Data Format

The input data should be a CSV file with the following columns:

- `gender`: Male/Female
- `SeniorCitizen`: 0/1
- `Partner`: Yes/No
- `Dependents`: Yes/No
- `tenure`: Number of months
- `PhoneService`: Yes/No
- `MultipleLines`: Yes/No/No phone service
- `InternetService`: DSL/Fiber optic/No
- `OnlineSecurity`: Yes/No/No internet service
- `OnlineBackup`: Yes/No/No internet service
- `DeviceProtection`: Yes/No/No internet service
- `TechSupport`: Yes/No/No internet service
- `StreamingTV`: Yes/No/No internet service
- `StreamingMovies`: Yes/No/No internet service
- `Contract`: Month-to-month/One year/Two year
- `PaperlessBilling`: Yes/No
- `PaymentMethod`: Electronic check/Mailed check/Bank transfer/Credit card
- `MonthlyCharges`: Float
- `TotalCharges`: String (will be converted to float)
- `Churn`: Yes/No (target variable, only for training)

## Models

The project implements and compares the following models:

1. **Logistic Regression**: Baseline linear model
2. **Random Forest**: Ensemble tree-based model
3. **XGBoost**: Gradient boosting model

Each model is tuned using GridSearchCV with cross-validation.

## Configuration

### Model Configuration (`config/model.yaml`)

Configure model hyperparameters and search grids for tuning.

### Schema Configuration (`config/schema.yaml`)

Define data schema, column types, and feature lists.

## Logging

All operations are logged to files in the `logs/` directory with timestamps. Logs include:
- Data ingestion steps
- Validation results
- Transformation details
- Model training metrics
- Prediction results

## Exception Handling

Custom exception handling provides detailed error messages including:
- File name where error occurred
- Line number
- Error description

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

Abhay Kumar

## Acknowledgments

- Scikit-learn for ML algorithms
- XGBoost for gradient boosting
- Flask for web framework
- Evidently for data drift detection