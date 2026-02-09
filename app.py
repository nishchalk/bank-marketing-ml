import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 – Streamlit Deployment")

MODEL_DIR = "model"
METRICS_PATH = f"{MODEL_DIR}/metrics.json"

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
    ["Model Performance", "Training Notebook", "CSV Upload Prediction", "Single Prediction"]
)

if page == "Model Performance":
    st.subheader("Evaluation Metrics & Confusion Matrix")
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        metrics_list = metrics_data.get("models", [])
        if metrics_list:
            cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
            rows = []
            for m in metrics_list:
                rows.append({
                    "Model": m["name"],
                    "Accuracy": round(m["Accuracy"], 4),
                    "AUC": round(m["AUC"], 4),
                    "Precision": round(m["Precision"], 4),
                    "Recall": round(m["Recall"], 4),
                    "F1": round(m["F1"], 4),
                    "MCC": round(m["MCC"], 4),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No metrics data in metrics.json.")
    except FileNotFoundError:
        st.warning("metrics.json not found. Run the training notebook to generate it.")
    st.markdown("---")
    st.caption("Confusion matrix and classification report are shown on **CSV Upload Prediction** when you upload a CSV that includes the target column `y` and run prediction.")

elif page == "Training Notebook":
    st.subheader("Model Training Notebook")
    # Read the HTML file content
    with open("Bank_Marketing_Assignment_2.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    # Display the HTML content
    st.components.v1.html(html_content, width=1200, height=1000, scrolling=True)
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
        has_target = "y" in df.columns

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

                # Confusion matrix and classification report (when target 'y' is present)
                st.markdown("---")
                st.subheader("Confusion Matrix & Classification Report")
                if has_target:
                    y_raw = df["y"]
                    # Support both yes/no and 0/1
                    if y_raw.dtype in (np.int64, np.float64, int, float):
                        y_true = np.asarray(y_raw).astype(int)
                    else:
                        y_true = y_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).values
                        if np.isnan(y_true).any():
                            y_true = y_raw.astype(int).values
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_true, preds)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                xticklabels=["Not Subscribed", "Subscribed"],
                                yticklabels=["Not Subscribed", "Subscribed"])
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"{model_name}")
                    st.pyplot(fig)
                    plt.close(fig)
                    st.write("**Classification Report**")
                    st.text(classification_report(y_true, preds, target_names=["Not Subscribed", "Subscribed"]))
                else:
                    st.info("Upload a CSV that includes the target column **y** (yes/no or 0/1) to see the confusion matrix and classification report here.")

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