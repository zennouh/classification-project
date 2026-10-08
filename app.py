from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "model" / "rf_model_951617.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Unable to load the prediction model: {error}")
    st.stop()


st.title("Customer Segmentation")
st.caption("Enter the customer's RFM values to identify their segment.")

with st.form("rfm_form"):
    st.subheader("Customer information")

    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=0.0,
        step=10.0,
        help="The customer's total spending.",
    )
    recency = st.number_input(
        "Recency",
        min_value=0,
        value=0,
        step=1,
        help="The number of days since the customer's last purchase.",
    )
    frequency = st.number_input(
        "Frequency",
        min_value=0,
        value=0,
        step=1,
        help="The number of purchases made by the customer.",
    )

    submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

if submitted:
    customer_data = pd.DataFrame(
        [
            {
                "frequence": frequency,
                "recency": recency,
                "montary": monetary,
            }
        ]
    )
    prediction = int(model.predict(customer_data)[0])
    segment_names = {0: "Low-value customer", 1: "Medium-value customer", 2: "High-value customer"}
    segment_name = segment_names.get(prediction, f"Customer segment {prediction} ({prediction})")

    st.success(f"Predicted segment: **{segment_name}**")

