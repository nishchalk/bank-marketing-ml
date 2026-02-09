import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 – Streamlit Deployment")

MODEL_DIR = "model"

# Categorical options (encoding order = sorted unique in training data)
# Used for both Single Prediction and CSV Upload preprocessing
JOB_OPTIONS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]
MARITAL_OPTIONS = ["divorced", "married", "single"]
EDUCATION_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
DEFAULT_OPTIONS = ["no", "yes"]
HOUSING_OPTIONS = ["no", "yes"]
LOAN_OPTIONS = ["no", "yes"]
CONTACT_OPTIONS = ["cellular", "telephone", "unknown"]
MONTH_OPTIONS = ["apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"]
POUTCOME_OPTIONS = ["failure", "other", "success", "unknown"]

FEATURE_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome",
]

CATEGORICAL_MAP = {
    "job": {v: i for i, v in enumerate(JOB_OPTIONS)},
    "marital": {v: i for i, v in enumerate(MARITAL_OPTIONS)},
    "education": {v: i for i, v in enumerate(EDUCATION_OPTIONS)},
    "default": {v: i for i, v in enumerate(DEFAULT_OPTIONS)},
    "housing": {v: i for i, v in enumerate(HOUSING_OPTIONS)},
    "loan": {v: i for i, v in enumerate(LOAN_OPTIONS)},
    "contact": {v: i for i, v in enumerate(CONTACT_OPTIONS)},
    "month": {v: i for i, v in enumerate(MONTH_OPTIONS)},
    "poutcome": {v: i for i, v in enumerate(POUTCOME_OPTIONS)},
}


def encode_csv_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns to match training pipeline. Returns df with FEATURE_COLUMNS order, all numeric."""
    df = df.copy()
    if "y" in df.columns:
        df = df.drop(columns=["y"])
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    for col, mapping in CATEGORICAL_MAP.items():
        unknown_idx = mapping.get("unknown", 0)
        df[col] = df[col].astype(str).str.strip().str.lower().map(lambda x: mapping.get(x, unknown_idx))
        if df[col].isna().any():
            df[col] = df[col].fillna(unknown_idx)
        df[col] = df[col].astype(np.int64)
    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)
    return df[FEATURE_COLUMNS]


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
    st.components.v1.html(html_content, width="stretch", height="stretch")
    # with open("Bank_Marketing_Assignment_2.ipynb", "r", encoding="utf-8") as f:
    #     st.code(f.read(), language="python")

elif page == "CSV Upload Prediction":
    st.subheader("Upload CSV for Batch Prediction")

    # Sample CSV download
    SAMPLE_CSV_PATH = "dataset/test.csv"
    try:
        with open(SAMPLE_CSV_PATH, "rb") as f:
            sample_csv_bytes = f.read()
        st.download_button(
            "Download sample CSV",
            data=sample_csv_bytes,
            file_name="sample_bank_marketing.csv",
            mime="text/csv",
        )
    except FileNotFoundError:
        pass  # sample file not present

    model_name = st.selectbox("Select Model", list(models.keys()))
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Run Prediction"):
            model = models[model_name]
            try:
                df_encoded = encode_csv_for_prediction(df)
            except Exception as e:
                st.error(f"Preprocessing failed: {e}. Ensure CSV has columns: {FEATURE_COLUMNS}")
            else:
                if model_name in ["Logistic Regression", "KNN"]:
                    df_scaled = scaler.transform(df_encoded)
                    preds = model.predict(df_scaled)
                else:
                    preds = model.predict(df_encoded)
                df["Prediction"] = preds
                st.success("Prediction completed")
                st.dataframe(df)

elif page == "Single Prediction":
    st.subheader("Single Entry Prediction")

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