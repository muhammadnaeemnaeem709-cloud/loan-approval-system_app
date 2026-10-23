import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval System", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.result-card {padding: 1.5rem; border-radius: 14px; text-align: center; color: white;}
.result-card h1 {color: white; margin: 0;}
.result-card p {margin: 0.3rem 0 0 0; font-size: 1.1rem;}
.approved {background: linear-gradient(135deg, #1e8449, #27ae60);}
.rejected {background: linear-gradient(135deg, #922b21, #e74c3c);}
</style>
""", unsafe_allow_html=True)

base = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_all():
    def load(name):
        return joblib.load(os.path.join(base, name))
    models = {
        "Logistic Regression": load("logistic_model.pkl"),
        "KNN": load("knn_model.pkl"),
        "Naive Bayes": load("naive_model.pkl"),
    }
    return (models, load("scaler.pkl"), load("label_encoder.pkl"),
            load("edu_encoder.pkl"), load("onehot_encoder.pkl"), load("meta.pkl"))

models, scaler, le, edu_enc, onehot, meta = load_all()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    choice = st.selectbox("Select Model", list(models.keys()))
    st.divider()
    st.caption("About")
    st.write("This app predicts whether a loan application is likely "
             "to be approved using machine learning.")
    st.caption("Built with Python, scikit-learn and Streamlit")

# ---------------- Header ----------------
st.title("🏦 Loan Approval Prediction")
st.write("Fill in the applicant details below and click **Predict**.")

# ---------------- Input form ----------------
inputs = {}
with st.form("loan_form"):
    st.subheader("📊 Financial & Personal Details")
    cols = st.columns(3)
    for i, c in enumerate(meta["num_cols"]):
        with cols[i % 3]:
            inputs[c] = st.number_input(c.replace("_", " "),
                                        value=float(meta["num_defaults"][c]))

    st.subheader("📋 Categories")
    cat_inputs = ["Education_Level"] + list(meta["cat_cols"])
    cat_options = [list(edu_enc.classes_)] + [list(x) for x in onehot.categories_]
    cols2 = st.columns(3)
    for i, (c, opts) in enumerate(zip(cat_inputs, cat_options)):
        with cols2[i % 3]:
            inputs[c] = st.selectbox(c.replace("_", " "), opts)

    submitted = st.form_submit_button("🔍 Predict", type="primary")

# ---------------- Prediction ----------------
if submitted:
    row = pd.DataFrame([inputs])

    row["Education_Level"] = edu_enc.transform(row["Education_Level"])
    enc = onehot.transform(row[meta["cat_cols"]])
    enc_df = pd.DataFrame(enc, columns=onehot.get_feature_names_out(meta["cat_cols"]))
    row = pd.concat([row.drop(columns=meta["cat_cols"]).reset_index(drop=True), enc_df], axis=1)

    row["Applicant_Income_sq"] = row["Applicant_Income"] ** 2
    row["DTI_Ratio_sq"] = row["DTI_Ratio"] ** 2
    row["Credit_Score_sq"] = row["Credit_Score"] ** 2
    row = row.drop(columns=["Applicant_Income", "DTI_Ratio", "Credit_Score"])
    row = row.reindex(columns=meta["feature_columns"], fill_value=0)

    scaled = scaler.transform(row)

    model = models[choice]
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    label = str(le.inverse_transform([pred])[0])
    approved = label.strip().lower() in ["yes", "y", "approved", "1"]

    # probability of "approved" class
    classes = list(le.classes_)
    yes_idx = next((i for i, c in enumerate(classes)
                    if str(c).strip().lower() in ["yes", "y", "approved", "1"]), 1)
    approve_prob = float(proba[yes_idx])

    st.divider()
    st.subheader("Result")

    if approved:
        st.markdown(f"""<div class="result-card approved">
            <h1>✅ Loan Approved</h1>
            <p>Model: {choice}</p></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-card rejected">
            <h1>❌ Loan Rejected</h1>
            <p>Model: {choice}</p></div>""", unsafe_allow_html=True)

    st.write("")
    m1, m2 = st.columns(2)
    m1.metric("Confidence", f"{proba.max():.0%}")
    m2.metric("Approval Probability", f"{approve_prob:.0%}")
    st.progress(approve_prob)
