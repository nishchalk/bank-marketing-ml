# bank-marketing-ml

Bank marketing classification project using the UCI Bank Marketing dataset. Predicts whether a customer subscribes to a term deposit. Includes model training (Jupyter notebook) and a Streamlit app for inference and deployment.

---

## a. Problem statement

Predict whether a bank client will subscribe to a **term deposit** (variable `y`: yes/no) after a direct marketing campaign (phone calls). This is a **binary classification** problem. The solution is implemented with multiple classifiers and deployed as an interactive Streamlit app.

---

## b. Dataset description

- **Source**: [Bank Marketing UCI – Kaggle](https://www.kaggle.com/competitions/bank-marketing-uci/data) (Moro et al., 2011).
- **Context**: Direct marketing campaigns (phone calls) of a Portuguese banking institution.
- **Goal**: Predict if the client will subscribe to a term deposit (`y`: yes/no).
- **Instances**: 4,521 (10% sample) or 45,211 (full); minimum assignment size (500+) satisfied.
- **Features**: 16 input attributes (≥12 required):
  - **Client**: age, job, marital, education, default, balance, housing, loan
  - **Campaign**: contact, day, month, duration, campaign, pdays, previous, poutcome
- **Target**: `y` (binary: yes / no).
- **Citation**: S. Moro, R. Laureano and P. Cortez. *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.* Proceedings of ESM'2011, Guimarães, Portugal, 2011. [hdl.handle.net/1822/14838](http://hdl.handle.net/1822/14838)  
- **Attribute details**: see `dataset/bank-names.txt`.

---

## c. Models used

### Comparison table (evaluation metrics)

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1     | MCC    |
|--------------------------|----------|-------|-----------|--------|--------|--------|
| Logistic Regression      | 0.8829   | 0.8545| 0.4750    | 0.1827 | 0.2639 | 0.2428 |
| Decision Tree            | 0.8564   | 0.6678| 0.3860    | 0.4231 | 0.4037 | 0.3226 |
| kNN                      | 0.8906   | 0.7434| 0.5714    | 0.1923 | 0.2878 | 0.2871 |
| Naive Bayes              | 0.8398   | 0.8211| 0.3360    | 0.4038 | 0.3668 | 0.2775 |
| Random Forest (Ensemble)  | 0.8884   | 0.9004| 0.5246    | 0.3077 | 0.3879 | 0.3453 |
| XGBoost (Ensemble)       | 0.8906   | 0.9132| 0.5325    | 0.3942 | 0.4530 | 0.3993 |

### Observations on model performance

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | High accuracy and good AUC; low recall for the positive class (subscription), so it is conservative and misses many subscribers. |
| Decision Tree            | Moderate accuracy and lowest AUC; more balanced precision/recall than logistic regression but still moderate overall. |
| kNN                      | Highest accuracy with K=5; good precision but low recall for positives, so many subscribers are missed. |
| Naive Bayes              | Lowest accuracy and precision; moderate recall. Fast but less suited to this feature mix without stronger calibration. |
| Random Forest (Ensemble)  | Strong AUC and good balance; better recall than logistic regression and kNN, with robust performance. |
| XGBoost (Ensemble)       | Best AUC and MCC; best F1 and recall among all models. Best overall discriminative performance for this dataset. |

---

## Project structure

- **`Bank_Marketing_Assignment_2.ipynb`** – Training notebook: EDA, preprocessing, and training of all six classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost).
- **`Bank_Marketing_Assignment_2.py`** – Python script export of the notebook.
- **`app.py`** – Streamlit app: training notebook view (HTML), CSV upload (batch prediction), single prediction, and model performance (metrics + confusion matrix).
- **`dataset/`** – Dataset files; `bank-names.txt` has attribute definitions and citation.
- **`model/`** – Saved models (`.pkl`), scaler, and metrics used by the app.

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
