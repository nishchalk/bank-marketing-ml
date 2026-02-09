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
    # Read the HTML file content
    with open("Bank_Marketing_Assignment_2.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    # Display the HTML content
    st.components.v1.html(html_content, width=800, height=600, scrolling=True)
    # with open("Bank_Marketing_Assignment_2.ipynb", "r", encoding="utf-8") as f:
    #     st.code(f.read(), language="python")

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

    # Categorical options from UCI Bank Marketing dataset (encoding order = sorted unique in data)
    JOB_OPTIONS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]
    MARITAL_OPTIONS = ["divorced", "married", "single"]
    EDUCATION_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
    DEFAULT_OPTIONS = ["no", "yes"]
    HOUSING_OPTIONS = ["no", "yes"]
    LOAN_OPTIONS = ["no", "yes"]
    CONTACT_OPTIONS = ["cellular", "telephone", "unknown"]
    MONTH_OPTIONS = ["apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"]
    POUTCOME_OPTIONS = ["failure", "other", "success", "unknown"]

    # Column order must match notebook: X = df.drop('y', axis=1) → 16 features
    FEATURE_COLUMNS = [
        "age", "job", "marital", "education", "default", "balance",
        "housing", "loan", "contact", "day", "month", "duration",
        "campaign", "pdays", "previous", "poutcome",
    ]

    age = st.number_input("Age", 18, 100, 35)
    job_idx = st.selectbox("Job", options=range(len(JOB_OPTIONS)), format_func=lambda i: JOB_OPTIONS[i], index=0)
    job = job_idx
    marital_idx = st.selectbox("Marital status", options=range(len(MARITAL_OPTIONS)), format_func=lambda i: MARITAL_OPTIONS[i], index=1)
    marital = marital_idx
    education_idx = st.selectbox("Education", options=range(len(EDUCATION_OPTIONS)), format_func=lambda i: EDUCATION_OPTIONS[i], index=1)
    education = education_idx
    default_idx = st.selectbox("Credit in default?", options=range(len(DEFAULT_OPTIONS)), format_func=lambda i: DEFAULT_OPTIONS[i], index=0)
    default = default_idx
    balance = st.number_input("Balance", value=1000)
    housing_idx = st.selectbox("Housing loan?", options=range(len(HOUSING_OPTIONS)), format_func=lambda i: HOUSING_OPTIONS[i], index=0)
    housing = housing_idx
    loan_idx = st.selectbox("Personal loan?", options=range(len(LOAN_OPTIONS)), format_func=lambda i: LOAN_OPTIONS[i], index=0)
    loan = loan_idx
    contact_idx = st.selectbox("Contact type", options=range(len(CONTACT_OPTIONS)), format_func=lambda i: CONTACT_OPTIONS[i], index=0)
    contact = contact_idx
    day = st.number_input("Day of Month", 1, 31, 15)
    month_idx = st.selectbox("Month", options=range(len(MONTH_OPTIONS)), format_func=lambda i: MONTH_OPTIONS[i], index=4)
    month = month_idx
    duration = st.number_input("Call Duration", value=180)
    campaign = st.number_input("Campaign Contacts", value=1)
    pdays = st.number_input("Days Since Last Contact", value=999)
    previous = st.number_input("Previous Contacts", value=0)
    poutcome_idx = st.selectbox("Previous campaign outcome", options=range(len(POUTCOME_OPTIONS)), format_func=lambda i: POUTCOME_OPTIONS[i], index=0)
    poutcome = poutcome_idx

    input_df = pd.DataFrame(
        [[
            age, job, marital, education, default, balance,
            housing, loan, contact, day, month, duration,
            campaign, pdays, previous, poutcome,
        ]],
        columns=FEATURE_COLUMNS,
    )

    if st.button("Predict"):
        rows = []
        for model_name, model in models.items():
            if model_name in ["Logistic Regression", "KNN"]:
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
            else:
                pred = model.predict(input_df)[0]
            result = "Subscribed" if pred == 1 else "Not Subscribed"
            rows.append({"Model": model_name, "Prediction": result})
        pred_df = pd.DataFrame(rows)
        st.success("Predictions from all models")
        st.dataframe(pred_df, use_container_width=True, hide_index=True)