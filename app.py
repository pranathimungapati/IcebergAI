
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="IcebergAI | Detection System",
    page_icon="🧊",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0b1220;
}
[data-testid="stMetric"] {
    background-color: #172337;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return joblib.load("iceberg_final_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("iceberg_order_dataset.csv")

model = load_model()
df = load_dataset()

FEATURES = [
    "initial_displayed_quantity",
    "average_displayed_quantity",
    "total_executed_quantity",
    "replenishment_count",
    "execution_display_ratio"
]

# ---------------- SIDEBAR ----------------

st.sidebar.title("🧊 IcebergAI")
st.sidebar.caption("Hidden Institutional Order Detection")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Detect Order", "Model Performance", "Dataset Explorer"]
)

st.sidebar.divider()
st.sidebar.caption("Mathematics Hackathon Project")

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("🧊 IcebergAI")
    st.subheader("AI-Based Detection of Hidden Institutional Orders")
    st.write(
        "An intelligent order-book analysis system that identifies "
        "potential iceberg orders using machine learning."
    )

    st.info(
        "Educational prototype: this project uses synthetic order-book "
        "data. Its predictions are not verified real-market detections "
        "or financial advice."
    )

    total_orders = len(df)
    iceberg_orders = int((df["label"] == 1).sum())
    normal_orders = int((df["label"] == 0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Orders", total_orders)
    c2.metric("Normal Orders", normal_orders)
    c3.metric("Iceberg Orders", iceberg_orders)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Order Class Distribution")
        counts = df["label"].value_counts().sort_index()
        chart_df = pd.DataFrame({
            "Order Type": ["Normal", "Iceberg"],
            "Count": [
                int(counts.get(0, 0)),
                int(counts.get(1, 0))
            ]
        })
        fig = px.pie(
            chart_df,
            names="Order Type",
            values="Count",
            hole=0.55
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("How IcebergAI Works")
    st.markdown("""
    1. **Input:** Order-book execution features.
    2. **Analysis:** Random Forest machine-learning model.
    3. **Prediction:** Normal Order or Potential Iceberg Order.
    4. **Signal:** Educational BUY, SELL or HOLD indication.
    """)

# ---------------- DETECT ORDER ----------------

elif page == "Detect Order":

    st.title("🔍 Iceberg Order Detection")
    st.write("Enter the order features to analyse a simulated order.")

    defaults = {
        feature: float(df[feature].median())
        for feature in FEATURES
    }

    with st.form("prediction_form"):

        st.subheader("Order Information")

        order_side = st.selectbox(
            "Order Side",
            ["BUY", "SELL"]
        )

        col1, col2 = st.columns(2)

        with col1:
            initial_qty = st.number_input(
                "Initial Displayed Quantity",
                min_value=0.0,
                value=defaults["initial_displayed_quantity"],
                step=1.0
            )

            average_qty = st.number_input(
                "Average Displayed Quantity",
                min_value=0.0,
                value=defaults["average_displayed_quantity"],
                step=1.0
            )

            total_executed = st.number_input(
                "Total Executed Quantity",
                min_value=0.0,
                value=defaults["total_executed_quantity"],
                step=1.0
            )

        with col2:
            replenishments = st.number_input(
                "Replenishment Count",
                min_value=0,
                value=int(defaults["replenishment_count"]),
                step=1
            )

            ratio = st.number_input(
                "Execution / Display Ratio",
                min_value=0.0,
                value=defaults["execution_display_ratio"],
                step=0.1
            )

        submitted = st.form_submit_button(
            "Analyse Order",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame([[
            initial_qty,
            average_qty,
            total_executed,
            replenishments,
            ratio
        ]], columns=FEATURES)

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        iceberg_index = list(model.classes_).index(1)
        iceberg_probability = float(probabilities[iceberg_index])

        st.divider()
        st.subheader("Analysis Result")

        a, b = st.columns(2)

        with a:
            if prediction == 1:
                st.error("Potential Iceberg Order Detected")
            else:
                st.success("Normal Order Predicted")

        with b:
            st.metric(
                "Iceberg Probability",
                f"{iceberg_probability * 100:.2f}%"
            )

        st.progress(iceberg_probability)

        if iceberg_probability >= 0.70:
            signal = order_side
            st.warning(
                f"Simulated Signal: {signal}"
            )
        else:
            signal = "HOLD"
            st.info("Simulated Signal: HOLD")

        st.caption(
            "This signal is generated from the model's probability "
            "and the selected order side. It is for demonstration only."
        )

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.title("📊 Model Performance")
    st.write("Final held-out test results of the sequential synthetic dataset.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", "93.33%")
    c2.metric("Signal Precision", "87.50%")
    c3.metric("Iceberg Coverage", "91.30%")

    st.divider()

    st.subheader("Classification Report")

    report_df = pd.DataFrame({
        "Class": ["Normal Order", "Iceberg Order"],
        "Precision": [0.97, 0.88],
        "Recall": [0.92, 0.96],
        "F1-Score": [0.94, 0.92],
        "Support": [37, 23]
    })

    st.dataframe(report_df, hide_index=True, use_container_width=True)

    st.subheader("Confusion Matrix")

    matrix = np.array([
        [34, 3],
        [1, 22]
    ])

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Predicted Normal", "Predicted Iceberg"],
        y=["Actual Normal", "Actual Iceberg"],
        text=matrix,
        texttemplate="%{text}",
        colorscale="Blues"
    ))

    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Signal Evaluation")
    st.write("Test orders: 60")
    st.write("BUY / SELL signals: 24")
    st.write("Correct iceberg signals: 21")
    st.write("Actual iceberg orders: 23")

    st.caption(
        "All reported results come from the project's held-out "
        "synthetic test set. They do not establish real-market performance."
    )

# ---------------- DATASET EXPLORER ----------------

elif page == "Dataset Explorer":

    st.title("📁 Dataset Explorer")

    st.write("Explore the 300-order synthetic dataset.")

    c1, c2 = st.columns(2)
    c1.metric("Dataset Rows", len(df))
    c2.metric("Dataset Columns", len(df.columns))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Feature Statistics")
    st.dataframe(
        df[FEATURES].describe().round(2),
        use_container_width=True
    )

    selected_feature = st.selectbox(
        "Select a feature",
        FEATURES
    )

    fig = px.histogram(
        df,
        x=selected_feature,
        color="label",
        barmode="overlay",
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Dataset CSV",
        data=df.to_csv(index=False),
        file_name="iceberg_order_dataset.csv",
        mime="text/csv"
    )

# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "IcebergAI | Mathematics Hackathon | "
    "Synthetic-data educational prototype"
)
