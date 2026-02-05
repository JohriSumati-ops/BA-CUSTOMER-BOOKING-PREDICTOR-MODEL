import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and features
# -----------------------------
model = joblib.load("booking_model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Flight Booking Predictor", layout="centered")

st.title("✈️ Flight Booking Completion Predictor")
st.write("Predict whether a customer will complete a flight booking")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Customer Details")

num_passengers = st.slider("Number of Passengers", 1, 9, 1)
purchase_lead = st.slider("Purchase Lead (days before flight)", 0, 365, 30)
length_of_stay = st.slider("Length of Stay (days)", 1, 60, 7)
flight_duration = st.slider("Flight Duration (hours)", 1.0, 20.0, 6.0)

sales_channel = st.selectbox("Sales Channel", ["Internet", "Mobile"])
trip_type = st.selectbox("Trip Type", ["RoundTrip", "OneWay"])
flight_day = st.selectbox("Flight Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

wants_extra_baggage = st.selectbox("Extra Baggage", [0, 1])
wants_preferred_seat = st.selectbox("Preferred Seat", [0, 1])
wants_in_flight_meals = st.selectbox("In-flight Meals", [0, 1])

# -----------------------------
# Simple Encoding
# (must match training logic)
# -----------------------------
sales_channel_enc = 1 if sales_channel == "Internet" else 0
trip_type_enc = 1 if trip_type == "RoundTrip" else 0
flight_day_enc = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(flight_day)

# -----------------------------
# Create input in correct order
# -----------------------------
input_dict = {
    "num_passengers": num_passengers,
    "sales_channel": sales_channel_enc,
    "trip_type": trip_type_enc,
    "purchase_lead": purchase_lead,
    "length_of_stay": length_of_stay,
    "flight_hour": 12,              # fixed default
    "flight_day": flight_day_enc,
    "route": 0,                     # default value
    "booking_origin": 0,            # default value
    "wants_extra_baggage": wants_extra_baggage,
    "wants_preferred_seat": wants_preferred_seat,
    "wants_in_flight_meals": wants_in_flight_meals,
    "flight_duration": flight_duration
}

input_df = pd.DataFrame([input_dict])

# Ensure correct feature order
input_df = input_df[features]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Booking Completion"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.success(f"✅ Booking Likely to Complete\n\nProbability: {probability:.2%}")
    else:
        st.error(f"❌ Booking Unlikely to Complete\n\nProbability: {probability:.2%}")
