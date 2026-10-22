import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
    return joblib.load(path)


model = load_model()

st.title("Heart Disease Prediction System")
st.write("Enter the patient's details and click Predict.")

age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Female", "Male"])
cp_options = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4,
}
cp_label = st.selectbox("Chest pain type", list(cp_options.keys()))
thalach = st.number_input("Maximum heart rate achieved", min_value=70, max_value=210, value=150)

if st.button("Predict"):
    sample = pd.DataFrame(
        [[age, 1 if sex == "Male" else 0, cp_options[cp_label], thalach]],
        columns=["age", "sex", "cp", "thalach"],
    )
    result = model.predict(sample)[0]
    if result == 1:
        st.error("Prediction: Heart Disease likely")
    else:
        st.success("Prediction: No Heart Disease")
    st.caption("This is a student project and not medical advice.")