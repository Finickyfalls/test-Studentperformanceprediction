
import streamlit as st
import joblib
import pandas as pd

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# LOAD MODEL AND SCALER
# These must be the exact same files saved in Cell 46, sitting in the same
# folder as app.py when deployed.
model = joblib.load("spp_best_model.joblib")
scaler = joblib.load("spp_scaler.joblib")

st.title("🎓 Student Performance Prediction")
st.write("Enter a student's study habits below to predict their total score (out of 100).")

# INPUT SECTION
# Sliders matching the exact features the model was trained on, in the same order.
weekly_self_study_hours = st.slider("Weekly self-study hours", 0.0, 40.0, 15.0, 0.5)
attendance_percentage = st.slider("Attendance percentage (%)", 50.0, 100.0, 85.0, 0.1)
class_participation = st.slider("Class participation score (0-10)", 0.0, 10.0, 6.0, 0.1)

# PREDICTION
if st.button("Predict Total Score"):
    # Build a single-row DataFrame with the SAME column names/order used in training
    input_df = pd.DataFrame([{
        'weekly_self_study_hours': weekly_self_study_hours,
        'attendance_percentage': attendance_percentage,
        'class_participation': class_participation
    }])

    # Apply the same scaler fitted during training — never fit a new one here
    input_scaled = input_df.copy()
    input_scaled[input_df.columns] = scaler.transform(input_df[input_df.columns])

    prediction = model.predict(input_scaled)[0]
    prediction = max(0, min(100, prediction))  # clip to a realistic 0-100 range

    st.success(f"Predicted Total Score: {prediction:.1f} / 100")

    # Bonus: rough letter-grade band based on the predicted score
    if prediction >= 90:
        st.write("Estimated grade band: **A**")
    elif prediction >= 80:
        st.write("Estimated grade band: **B**")
    elif prediction >= 70:
        st.write("Estimated grade band: **C**")
    elif prediction >= 60:
        st.write("Estimated grade band: **D**")
    else:
        st.write("Estimated grade band: **F**")
