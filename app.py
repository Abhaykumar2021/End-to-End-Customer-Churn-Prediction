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
    
    # 1. Create Dictionary
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
    
    # 2. Create DataFrame
    pred_df = pd.DataFrame(input_dict)
    
    # 3. DEBUG: Force Data Types (The Nuclear Fix)
    # We explicitly tell Pandas: "These are numbers, and these are strings."
    try:
        # Force Numerics
        pred_df["Tenure Months"] = pd.to_numeric(pred_df["Tenure Months"], errors='coerce')
        pred_df["Monthly Charges"] = pd.to_numeric(pred_df["Monthly Charges"], errors='coerce')
        pred_df["Total Charges"] = pd.to_numeric(pred_df["Total Charges"], errors='coerce')
        
        # Force Categoricals to String (prevents weird object types)
        # Note: If your pipeline expects 'Senior Citizen' as 0/1 (int), remove it from this list.
        # But usually, it's safer to treat it as categorical if you OHE'd it.
        categorical_cols = [
            "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Multiple Lines", 
            "Internet Service", "Online Security", "Online Backup", "Device Protection", 
            "Tech Support", "Streaming TV", "Streaming Movies", "Contract", 
            "Paperless Billing", "Payment Method"
        ]
        
        # Check if columns exist before casting (handles potential naming mismatches)
        for col in categorical_cols:
            if col in pred_df.columns:
                pred_df[col] = pred_df[col].astype(str)

    except Exception as e:
        st.error(f"Data Type Conversion Error: {e}")
        st.stop()
    
    # 4. DEBUG DISPLAY (Remove this after it works)
    # This will show you exactly what Pandas thinks your data is. 
    # Look for 'object' where it should be 'float64'.
    with st.expander("Debug: View Data Types"):
        st.write(pred_df.dtypes)
        st.write(pred_df)

    try:
        # 5. Transform
        processed_data = preprocessor.transform(pred_df)
        
        # 6. Predict
        prediction = model.predict(processed_data)
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