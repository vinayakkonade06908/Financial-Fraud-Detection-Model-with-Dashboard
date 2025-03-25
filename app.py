import streamlit as st
import requests

st.title("Financial Fraud Detection Dashboard")

# Collect user inputs
amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
transaction_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])
merchant_category = st.selectbox("Merchant Category", ["Retail", "Electronics", "Food", "Clothing", "Groceries", "Luxury", "Travel","Restaurants", "Pharmaccy"])
transaction_count = st.slider("Transaction Count", 1, 1001)

if st.button("Predict Fraud"):
    # Prepare JSON payload
    data = {
        "amount": amount,
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "transaction_count": transaction_count
    }

    st.write("Sending data:", data)  # Debugging: Check if data is correct

    try:
        response = requests.post("http://127.0.0.1:8000/predict/", json=data)
        st.write("Response Status Code:", response.status_code)
        st.write("Response Text:", response.text)

        # Attempt to decode JSON response
        result = response.json()
        if "fraud" in result:
            if result["fraud"]:
                st.error("⚠ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Transaction is Safe")
        else:
            st.error("Error: Invalid response from server.")
    except requests.exceptions.JSONDecodeError:
        st.error("Error: Could not decode JSON response from server.")