import streamlit as st
import numpy as np
import pandas as pd
import json
import os
from xgboost import XGBClassifier

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_model.json")
model = XGBClassifier()
model.load_model('MODEL_PATH')
 

## streamlit app
st.title('Traffic Congestion Incident Detection Project')

# User input
vehicle_density = st.number_input('Vehicle Density')
avg_vehicle_speed = st.number_input('Avg Vehicle Speed')
speed_std = st.number_input('Speed Std')
lane_occupancy = st.number_input('Lane Occupancy')
queue_length = st.number_input('Queue Length')
edge_density = st.number_input('Edge Density')
optical_flow_mag = st.number_input('Optical Flow Mag')
shadow_fraction = st.number_input('Shadow Fraction')
time_of_day_norm = st.number_input('Time of Day Norm')
road_width_norm = st.number_input('Road Width Norm')


# Prepare the input data
input_data = pd.DataFrame({
    "vehicle_density":[vehicle_density],
    "avg_vehicle_speed":[avg_vehicle_speed],
    "speed_std":[speed_std],
    "lane_occupancy":[lane_occupancy],
    "queue_length":[queue_length],
    "edge_density":[edge_density],
    "optical_flow_mag":[optical_flow_mag],
    "shadow_fraction":[shadow_fraction],
    "time_of_day_norm":[time_of_day_norm],
    "road_width_norm":[road_width_norm]
})


if st.button("Predict Congestion And Risk Probabilities"):
    pred_proba = model.predict_proba(input_data)[:, 1][0]
    st.metric(
        label="Congestion Risk",
        value=f"{pred_proba:.3f}"
    )

    if pred_proba >= 0.5:
        st.write('Congestion With Very High Risk.')
    elif pred_proba>= 0.25:
        st.write('Congestion With Very Mild Risk.')
    else:
        st.write('Congestion Not Likely.') 
