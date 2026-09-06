import streamlit as st
import requests

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction feature values to predict fraud probability.")

# Input fields for key features
# Key features including the heavy anomaly indicators
v1 = st.number_input("V1", value=-2.3122, format="%.4f")
v10 = st.number_input("V10 (Critical Indicator)", value=-5.5343, format="%.4f")
v12 = st.number_input("V12 (Critical Indicator)", value=-6.1554, format="%.4f")
v14 = st.number_input("V14 (Strongest Fraud Signal)", value=-9.2115, format="%.4f")
v17 = st.number_input("V17 (Critical Indicator)", value=-9.5243, format="%.4f")
amount = st.number_input("Amount ($)", value=0.00, format="%.2f")

# Pointing directly to your live Render backend
API_URL = "https://xg-boost-gdwv.onrender.com/predict"  

if st.button("Analyze Transaction"):
    payload = {
        "features": {
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "Amount": amount
        }
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["fraud_probability"] * 100
            
            if result["is_fraud"]:
                st.error(f"⚠️ High Risk: Fraud Detected! (Probability: {prob:.2f}%)")
            else:
                st.success(f"✅ Low Risk: Legitimate Transaction (Probability: {prob:.2f}%)")
        else:
            st.error(f"Render Error ({response.status_code}): {response.text}")
            
    except Exception as e:
        st.error(f"Connection failed entirely: {e}")
