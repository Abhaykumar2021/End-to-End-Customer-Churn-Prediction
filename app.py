import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1. APP CONFIGURATION
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# 2. LOAD ARTIFACTS
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

# 3. UI LAYOUT
st.title("📉 Customer Churn Prediction System")
st.markdown("Enter customer details below to predict churn probability.")

tab1, tab2, tab3 = st.tabs(["👤 Demographics", "📞 Services", "💰 Account Info"])

# --- TAB 1: DEMOGRAPHICS ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

# --- TAB 2: SERVICES ---
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

# --- TAB 3: ACCOUNT INFO ---
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

# 4. PREDICTION LOGIC
if st.button("Predict Churn Status", type="primary", use_container_width=True):
    
    # Create input dictionary
    # CRITICAL: Ensure these keys match your training columns EXACTLY
    input_dict = {
        "gender": [gender],
        "SeniorCitizen": [senior_citizen], 
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    }
    
    pred_df = pd.DataFrame(input_dict)
    
    try:
        processed_data = preprocessor.transform(pred_df)
        prediction = model.predict(processed_data)
        score = prediction[0][0]
        
        st.divider()
        if score > 0.5:
            st.error(f"🚨 **CHURN PREDICTED** (Probability: {score:.2%})")
        else:
            st.success(f"✅ **NO CHURN** (Probability: {score:.2%})")
            
    except Exception as e:
        st.error(f"Error: {e}")