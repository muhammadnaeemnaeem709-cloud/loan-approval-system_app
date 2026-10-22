import os
import joblib
import pandas as pd
import streamlit as st

base = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return joblib.load(os.path.join(base, name))

models = {
    "Logistic Regression": load("logistic_model.pkl"),
    "KNN": load("knn_model.pkl"),
    "Naive Bayes": load("naive_model.pkl"),
}
scaler = load("scaler.pkl")
le = load("label_encoder.pkl")
edu_enc = load("edu_encoder.pkl")
onehot = load("onehot_encoder.pkl")
meta = load("meta.pkl")

st.set_page_config(page_title="Loan Approval System")
st.title("Loan Approval Prediction")

choice = st.selectbox("Select Model", list(models.keys()))

st.subheader("Applicant Details")
inputs = {}

for c in meta["num_cols"]:
    inputs[c] = st.number_input(c.replace("_", " "),
                                value=float(meta["num_defaults"][c]))

inputs["Education_Level"] = st.selectbox("Education Level",
                                         list(edu_enc.classes_))

for c, cats in zip(meta["cat_cols"], onehot.categories_):
    inputs[c] = st.selectbox(c.replace("_", " "), list(cats))

if st.button("Predict"):
    row = pd.DataFrame([inputs])

    # same encoding jo training mein ki thi
    row["Education_Level"] = edu_enc.transform(row["Education_Level"])
    enc = onehot.transform(row[meta["cat_cols"]])
    enc_df = pd.DataFrame(enc, columns=onehot.get_feature_names_out(meta["cat_cols"]))
    row = pd.concat([row.drop(columns=meta["cat_cols"]).reset_index(drop=True), enc_df], axis=1)

    # same feature engineering
    row["Applicant_Income_sq"] = row["Applicant_Income"] ** 2
    row["DTI_Ratio_sq"] = row["DTI_Ratio"] ** 2
    row["Credit_Score_sq"] = row["Credit_Score"] ** 2
    row = row.drop(columns=["Applicant_Income", "DTI_Ratio", "Credit_Score"])

    # columns ka order bilkul training jaisa
    row = row.reindex(columns=meta["feature_columns"], fill_value=0)

    scaled = scaler.transform(row)

    model = models[choice]
    pred = model.predict(scaled)[0]
    label = str(le.inverse_transform([pred])[0])
    confidence = model.predict_proba(scaled)[0].max()

    if label.strip().lower() in ["yes", "y", "approved", "1"]:
        st.success(f"Loan Approved (confidence: {confidence:.0%})")
    else:
        st.error(f"Loan Rejected (confidence: {confidence:.0%})")