# bank-marketing-ml

Bank marketing classification project using the UCI Bank Marketing dataset. Predicts whether a customer subscribes to a term deposit. Includes model training (Jupyter notebook) and a Streamlit app for inference.

## Dataset

- **Source**: [Bank Marketing UCI – Kaggle](https://www.kaggle.com/competitions/bank-marketing-uci/data)
- **Description**: Direct marketing campaigns (phone calls) of a Portuguese banking institution. Goal is to predict if the client will subscribe to a term deposit (target `y`: yes/no).
- **Instances**: 45,211 (full) or 4,521 (10% sample).
- **Attributes**: 16 input features (e.g. age, job, marital, education, balance, contact, duration, campaign, pdays, poutcome) + binary target.
- **Full attribute list, citation and details**: see **`dataset/bank-names.txt`** (from [Moro et al., 2011](http://hdl.handle.net/1822/14838)).

## Project structure

- **`Bank_Marketing_Assignment_2.ipynb`** – Training notebook: EDA, preprocessing, and training of multiple classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost).
- **`Bank_Marketing_Assignment_2.py`** – Python script export of the notebook.
- **`app.py`** – Streamlit app: view training notebook (HTML), batch prediction (CSV upload), and single prediction.
- **`bank_marketing.csv`** – Dataset.
- **`dataset/bank-names.txt`** – Dataset description, attribute definitions, and citation.
- **`model/`** – Saved models (`.pkl`) and scaler used by the app.

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Regenerate the HTML version of the notebook for the “Training Notebook” view in the app:

   ```bash
   jupyter nbconvert --to html Bank_Marketing_Assignment_2.ipynb
   ```

   This produces `Bank_Marketing_Assignment_2.html`, which the Streamlit app embeds.

## Run the Streamlit app

From the project root:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Requirements

See `requirements.txt`. Main dependencies: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`.
