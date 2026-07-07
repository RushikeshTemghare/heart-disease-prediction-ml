import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------
# Load Machine Learning Model Files
# ---------------------------------------

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered"
)


# ---------------------------------------
# App Introduction
# ---------------------------------------

st.title("❤️ Heart Disease Prediction App")

st.markdown(
    """
    ## Machine Learning-Based Cardiovascular Risk Assessment

    This application uses a **K-Nearest Neighbors (KNN) machine learning model**
    to predict the likelihood of heart disease based on patient health indicators.

    The model evaluates important cardiovascular features such as:

    - Age
    - Gender
    - Blood pressure
    - Cholesterol level
    - Blood sugar levels
    - Maximum heart rate
    - Exercise-induced angina
    - ECG results
    - ST depression and slope values

    Enter the patient's information below and click **Predict** to receive
    a heart disease risk assessment.
    """
)


st.info(
    """
    **Medical Disclaimer**

    This application is created for educational and demonstration purposes only.
    The prediction should not be considered a medical diagnosis or a replacement
    for professional healthcare advice. Always consult a qualified healthcare
    professional for medical decisions.
    """
)


st.markdown("---")


# ---------------------------------------
# Patient Information Section
# ---------------------------------------

st.subheader("🩺 Patient Information")

st.write(
    "Please provide the following health details:"
)


age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)


sex = st.selectbox(
    "Sex",
    ["M", "F"]
)


chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)


resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)


cholesterol = st.number_input(
    "Cholesterol Level (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)


fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)


resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)


max_hr = st.slider(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)


exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)


oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=6.0,
    value=1.0
)


st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)



# ---------------------------------------
# Prediction Section
# ---------------------------------------

st.markdown("---")

st.subheader("📊 Heart Disease Risk Prediction")


if st.button("🔍 Predict"):


    # Create user input dictionary

    raw_input = {

        "Age": age,

        "RestingBP": resting_bp,

        "Cholesterol": cholesterol,

        "FastingBS": fasting_bs,

        "MaxHR": max_hr,

        "Oldpeak": oldpeak,


        ("Sex_" + sex): 1,

        ("ChestPainType_" + chest_pain): 1,

        ("RestingECG_" + resting_ecg): 1,

        ("ExerciseAngina_" + exercise_angina): 1,

        ("ST_Slope_" + st_slope): 1
    }



    # Convert input into dataframe

    input_df = pd.DataFrame([raw_input])


    # Add missing training columns

    for col in expected_columns:

        if col not in input_df.columns:

            input_df[col] = 0



    # Arrange columns in correct order

    input_df = input_df[expected_columns]



    # Apply scaling

    scaled_input = scaler.transform(input_df)



    # Generate prediction

    prediction = model.predict(scaled_input)[0]



    # Display prediction result

    if prediction == 1:

        st.error(
            "⚠️ High Risk of Heart Disease"
        )

        st.write(
            "The model identified patterns associated with a higher probability of heart disease."
        )


    else:

        st.success(
            "✅ Low Risk of Heart Disease"
        )

        st.write(
            "The model identified patterns associated with a lower probability of heart disease."
        )



# ---------------------------------------
# Footer
# ---------------------------------------

st.markdown("---")

st.caption(
    "Built with Python, Scikit-Learn, KNN Algorithm and Streamlit | Machine Learning Portfolio Project"
)