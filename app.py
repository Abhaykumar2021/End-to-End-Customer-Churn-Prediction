import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

from src.pipeline.prediction_pipeline import PredictPipeline

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model('artifacts/model.h5')
        preprocessor = joblib.load('artifacts/preprocessor.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None

model, preprocessor = load_artifacts()

if model is None or preprocessor is None:
    st.stop()

st.title("📉 Customer Churn Prediction System")
st.markdown("Enter customer details below to predict churn probability.")

tab1, tab2, tab3 = st.tabs(["👤 Demographics", "📞 Services", "💰 Account Info"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col2:
        online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    with col3:
        tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col2:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges", min_value=18.0, max_value=120.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=monthly_charges * tenure)

    input_dict = {
        "Gender": [gender],
        "Senior Citizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure Months": [tenure],
        "Phone Service": [phone_service],
        "Multiple Lines": [multiple_lines],
        "Internet Service": [internet_service],
        "Online Security": [online_security],  
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Paperless Billing": [paperless_billing],
        "Payment Method": [payment_method],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges]
    }
    
    pred_df = pd.DataFrame(input_dict)
    
    try:
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        
        score = prediction[0][0]
        
        st.divider()
        if score > 0.5:
            st.error(f"🚨 **CHURN PREDICTED** (Probability: {score:.2%})")
            st.write("This customer is at high risk of leaving.")
        else:
            st.success(f"✅ **NO CHURN** (Probability: {score:.2%})")
            st.write("This customer is likely to stay.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning("If the error is 'columns are missing', check the keys in 'input_dict' again.")