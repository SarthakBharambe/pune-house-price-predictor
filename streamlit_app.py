# streamlit_app.py

import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open("pune_price_model.pkl", "rb"))

# Streamlit page setup
st.set_page_config(page_title="Pune House Price Predictor", layout="centered")
st.title("🏠 Pune House Price Prediction App")
st.markdown("Enter the property details to estimate the price")

# User input sliders
total_sqft = st.slider("Total Square Feet Area", min_value=300, max_value=10000, value=1000, step=50)
bhk = st.selectbox("BHK (Bedrooms)", [1, 2, 3, 4, 5])
bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])

# Predict Button
if st.button("Predict Price"):
    input_data = np.array([[total_sqft, bhk, bath]])
    predicted_price = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹ {round(predicted_price, 2)} Lakhs")
    st.balloons()
