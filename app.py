
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler with full paths
best_model = joblib.load('/content/spp_best_model.joblib')
scaler = joblib.load('/content/spp_scaler.joblib')

# Define the order of columns for prediction (must match X_train)
# Based on X_train.columns from the notebook state
feature_cols = [
    'weekly_self_study_hours',
    'attendance_percentage',
    'class_participation',
    'grade_B',
    'grade_C',
    'grade_D',
    'grade_F'
]

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title('Student Performance Prediction (SPP)')
st.write('Predict a student\'s total score based on their study habits, attendance, and class participation.')

st.subheader('Enter Student Details:')

# Input widgets for numerical features
weekly_self_study_hours = st.slider('Weekly Self-Study Hours', min_value=0.0, max_value=40.0, value=15.0, step=0.1)
attendance_percentage = st.slider('Attendance Percentage', min_value=50.0, max_value=100.0, value=85.0, step=0.1)
class_participation = st.slider('Class Participation (out of 10)', min_value=0.0, max_value=10.0, value=6.0, step=0.1)

# Input widget for categorical 'grade'
grade_options = ['A', 'B', 'C', 'D', 'F']
selected_grade = st.radio('Previous Grade (for one-hot encoding)', grade_options, index=0) # Default to 'A'

# Create a dictionary for one-hot encoded grade columns
grade_encoded = {
    'grade_B': 0,
    'grade_C': 0,
    'grade_D': 0,
    'grade_F': 0
}

# Set the corresponding grade column to 1 based on selection
if selected_grade != 'A':
    grade_encoded[f'grade_{selected_grade}'] = 1

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'weekly_self_study_hours': [weekly_self_study_hours],
    'attendance_percentage': [attendance_percentage],
    'class_participation': [class_participation],
    'grade_B': [grade_encoded['grade_B']],
    'grade_C': [grade_encoded['grade_C']],
    'grade_D': [grade_encoded['grade_D']],
    'grade_F': [grade_encoded['grade_F']]
})

# Identify numerical columns for scaling (these must match 'cols_to_scale' from training)
# Based on `cols_to_scale` variable from the notebook state
numerical_features_for_scaling = [
    'weekly_self_study_hours',
    'attendance_percentage',
    'class_participation'
]

# Scale the numerical features using the loaded scaler
input_data[numerical_features_for_scaling] = scaler.transform(input_data[numerical_features_for_scaling])

# Ensure column order matches the training data (important for prediction)
input_data = input_data[feature_cols]

if st.button('Predict Total Score'):
    prediction = best_model.predict(input_data)[0]
    st.success(f'Predicted Total Score: **{prediction:.2f}**')
    st.balloons()

st.markdown("""
--- 
#### How to run this application:
1.  Ensure `spp_best_model.joblib` and `spp_scaler.joblib` are in the `/content/` directory or update paths in `app.py`.
2.  Open your terminal or command prompt.
3.  Navigate to the directory where `app.py` is saved.
4.  Run the command: `streamlit run app.py`
""")
