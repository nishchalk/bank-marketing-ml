import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 – Streamlit Deployment")

MODEL_DIR = "model"

models = {
    "Logistic Regression": joblib.load(f"{MODEL_DIR}/logistic_regression.pkl"),
    "Decision Tree": joblib.load(f"{MODEL_DIR}/decision_tree.pkl"),
    "KNN": joblib.load(f"{MODEL_DIR}/knn.pkl"),
    "Naive Bayes": joblib.load(f"{MODEL_DIR}/naive_bayes.pkl"),
    "Random Forest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
    "XGBoost": joblib.load(f"{MODEL_DIR}/xgboost.pkl"),
}

scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

page = st.sidebar.radio(
    "Navigation",
    ["Training Notebook", "CSV Upload Prediction", "Single Prediction"]
)

if page == "Training Notebook":
    st.subheader("Model Training Notebook")
    with open("Bank_Marketing_Assignment_2.ipynb", "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")

elif page == "CSV Upload Prediction":
    st.subheader("Upload CSV for Batch Prediction")

    model_name = st.selectbox("Select Model", list(models.keys()))
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Run Prediction"):
            model = models[model_name]

            if model_name in ["Logistic Regression", "KNN"]:
                df_scaled = scaler.transform(df)
                preds = model.predict(df_scaled)
            else:
                preds = model.predict(df)

            df["Prediction"] = preds
            st.success("Prediction completed")
            st.dataframe(df)

elif page == "Single Prediction":
    st.subheader("Single Entry Prediction")

    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    age = st.number_input("Age", 18, 100, 35)
    balance = st.number_input("Balance", value=1000)
    duration = st.number_input("Call Duration", value=180)
    campaign = st.number_input("Campaign Contacts", value=1)
    pdays = st.number_input("Days Since Last Contact", value=999)
    previous = st.number_input("Previous Contacts", value=0)

    job = st.selectbox("Job (encoded)", list(range(12)))
    education = st.selectbox("Education (encoded)", list(range(4)))
    marital = st.selectbox("Marital (encoded)", list(range(3)))

    input_df = pd.DataFrame([[
        age, job, marital, education, balance,
        campaign, pdays, previous, duration
    ]])

    if st.button("Predict"):
        if model_name in ["Logistic Regression", "KNN"]:
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
        else:
            pred = model.predict(input_df)[0]

        result = "Subscribed" if pred == 1 else "Not Subscribed"
        st.success(f"Prediction: {result}")