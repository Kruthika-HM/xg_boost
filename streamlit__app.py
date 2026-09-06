import streamlit as st
import requests

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction feature values to predict fraud probability.")

# Input fields for key features
v1 = st.number_input("V1", value=-3.0435, format="%.4f")
v2 = st.number_input("V2", value=-2.1581, format="%.4f")
v3 = st.number_input("V3", value=-4.0123, format="%.4f")
v4 = st.number_input("V4", value=3.4561, format="%.4f")
amount = st.number_input("Amount ($)", value=1.00, format="%.2f")

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
