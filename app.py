import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean Classification — Model Evaluation")
st.write("Upload a labelled test CSV containing the 16 input features and the `Class` target column.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

available = {name: path for name, path in MODEL_FILES.items() if os.path.exists(path)}
if not available:
    st.error("No saved models found. Run the training notebook first and upload the model/ folder to GitHub.")
    st.stop()

selected = st.selectbox("Select a trained model", list(available.keys()))
uploaded = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)
    if "Class" not in data.columns:
        st.error("The uploaded CSV must include the target column named `Class`.")
        st.stop()

    model = joblib.load(available[selected])
    X_test = data.drop(columns=["Class"])
    y_test = data["Class"]

    expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    if expected is not None:
        missing = sorted(set(expected) - set(X_test.columns))
        extra = sorted(set(X_test.columns) - set(expected))
        if missing or extra:
            st.error(f"Feature mismatch. Missing: {missing}; extra: {extra}")
            st.stop()
        X_test = X_test[expected]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC (macro OVR)": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    st.subheader(f"Evaluation — {selected}")
    cols = st.columns(3)
    for i, (name, value) in enumerate(metrics.items()):
        cols[i % 3].metric(name, f"{value:.4f}")

    st.subheader("Confusion Matrix")
    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    output = data.copy()
    output["Predicted_Class"] = y_pred
    st.download_button("Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv")
else:
    st.info("Upload the test_data.csv created by the training notebook to evaluate the selected saved model.")
