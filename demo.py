from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

sample_customer = CustomData(
    gender="Female",
    senior_citizen=0,
    partner="No",
    dependents="No",
    tenure=2, 
    phone_service="Yes",
    multiple_lines="No",
    internet_service="Fiber optic", 
    online_security="No",
    online_backup="No",
    device_protection="No",
    tech_support="No",
    streaming_tv="No",
    streaming_movies="No",
    contract="Month-to-month",  
    paperless_billing="Yes",
    payment_method="Electronic check", 
    monthly_charges=70.0,
    total_charges=140.0
)

pred_df = sample_customer.get_data_as_data_frame()
print("Sample Customer Data:")
print(pred_df.T)

predict_pipeline = PredictPipeline()
print("\nRunning Prediction...")
results = predict_pipeline.predict(pred_df)

print(f"\nPrediction Probability: {results[0][0]}")
if results[0][0] > 0.5:
    print("Result: CHURN (The model predicts this customer will leave)")
else:
    print("Result: NO CHURN (The model predicts this customer will stay)")
