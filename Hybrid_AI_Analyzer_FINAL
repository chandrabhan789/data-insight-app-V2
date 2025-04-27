import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from pandas_profiling import ProfileReport
from pycaret.classification import *
from pycaret.clustering import *
from pycaret.time_series import *
from pycaret.nlp import *
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_pandas_profiling import st_profile_report
import io

# ======================
# 1. Setup: Load/Save Custom Insights
# ======================
INSIGHTS_FILE = "saved_insights.json"

def load_insights():
    if Path(INSIGHTS_FILE).exists():
        with open(INSIGHTS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_insights(insights):
    with open(INSIGHTS_FILE, "w") as f:
        json.dump(insights, f)

# ======================
# 2. Streamlit App
# ======================
st.set_page_config(page_title="Hybrid AI Analyzer", layout="wide")
st.title("🧠 Hybrid AI Analyzer")
st.markdown("Analyze **any data** + **teach AI custom insights**!")

# ----- Data Input -----
input_method = st.radio("Input Method:", ("📋 Paste Data", "📂 Upload File"))
data = None

if input_method == "📋 Paste Data":
    raw_text = st.text_area("Paste CSV/JSON:", height=200,
                          placeholder="CSV Example:\nDay,Sales\nMon,100\nTue,150\nJSON Example:\n[{\"Day\":\"Mon\",\"Sales\":100}]")
    if st.button("Parse Data") and raw_text:
        try:
            data = pd.read_json(io.StringIO(raw_text)) if raw_text.strip()[0] in ['{', '['] else pd.read_csv(io.StringIO(raw_text))
            st.success("✅ Data parsed!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    uploaded_file = st.file_uploader("Upload Data (CSV/Excel/JSON)", type=["csv", "xlsx", "json"])
    if uploaded_file:
        try:
            data = pd.read_json(uploaded_file) if uploaded_file.name.endswith('.json') else \
                  pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success("✅ Data loaded!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

if data is not None:
    st.dataframe(data.head(3))

    # ======================
    # 3. Universal Analysis (Auto-Detection)
    # ======================
    st.header("🔍 Auto-Analysis")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    text_cols = data.select_dtypes(include='object').columns.tolist()
    date_cols = data.select_dtypes(include='datetime').columns.tolist()

    # ----- Dynamic Tabs -----
    tab_names = []
    if numeric_cols:
        tab_names.extend(["📈 Stats & Correlations", "📊 Clustering"])
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        tab_names.append("⏳ Time-Series Forecast")
    if text_cols:
        tab_names.append("📝 NLP Analysis")
    if "target" in data.columns:
        tab_names.append("🎯 Classification")

    tabs = st.tabs(tab_names)

    # A. Stats & Correlations
    if "📈 Stats & Correlations" in tab_names:
        with tabs[tab_names.index("📈 Stats & Correlations")]:
            st.subheader("Descriptive Stats")
            st.write(data[numeric_cols].describe())
            st.subheader("Correlation Heatmap")
            corr = data[numeric_cols].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            st.pyplot(plt.gcf())

    # B. Clustering
    if "📊 Clustering" in tab_names:
        with tabs[tab_names.index("📊 Clustering")]:
            st.subheader("Cluster Analysis")
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            if st.button("Run Clustering"):
                with st.spinner("Clustering..."):
                    setup(data, normalize=True, session_id=123)
                    kmeans = create_model('kmeans', num_clusters=n_clusters)
                    data['Cluster'] = predict_model(kmeans, data=data)['Cluster']
                    st.success("Clustering done!")
                    st.dataframe(data.head())

    # C. Time-Series Forecast
    if "⏳ Time-Series Forecast" in tab_names:
        with tabs[tab_names.index("⏳ Time-Series Forecast")]:
            st.subheader("Forecasting")
            target_col = st.selectbox("Target Column", numeric_cols)
            forecast_period = st.slider("Forecast Periods", 1, 365, 7)
            if st.button("Run Forecast"):
                with st.spinner("Training..."):
                    setup(data, target=target_col, fh=forecast_period, session_id=123)
                    best_model = compare_models()
                    plot_model(best_model, plot='forecast')
                    st.pyplot(plt.gcf())

    # D. NLP Analysis
    if "📝 NLP Analysis" in tab_names:
        with tabs[tab_names.index("📝 NLP Analysis")]:
            st.subheader("Text Analysis")
            text_col = st.selectbox("Text Column", text_cols)
            if st.button("Analyze Text"):
                with st.spinner("Processing..."):
                    setup(data=data, target=text_col, session_id=123)
                    lda = create_model('lda')
                    plot_model(lda, plot='topic_distribution')
                    st.pyplot(plt.gcf())

    # E. Classification
    if "🎯 Classification" in tab_names:
        with tabs[tab_names.index("🎯 Classification")]:
            st.subheader("Predict Target")
            if st.button("Train Classifier"):
                with st.spinner("Training..."):
                    setup(data, target="target", session_id=123)
                    best_model = compare_models()
                    plot_model(best_model, plot='confusion_matrix')
                    st.pyplot(plt.gcf())

    # ======================
    # 4. Trainable Insights
    # ======================
    st.header("🎓 Teach AI Custom Insights")
    insights = load_insights()

    # A. Add New Insight
    new_insight_name = st.text_input("Name this insight (e.g., 'Weekend Sales Drop'):")
    new_insight_logic = st.text_area("Logic (Python code using `data`):",
                                   placeholder="e.g., data[data['Day'].isin(['Sat','Sun'])]['Sales'].mean() < data[~data['Day'].isin(['Sat','Sun'])]['Sales'].mean()")

    if st.button("💾 Save Insight") and new_insight_name and new_insight_logic:
        insights[new_insight_name] = new_insight_logic
        save_insights(insights)
        st.success(f"Saved: '{new_insight_name}'!")

    # B. Apply Learned Insights
    st.subheader("🔎 Your Custom Insights")
    for name, logic in insights.items():
        try:
            result = eval(logic, {'data': data})
            st.write(f"✅ **{name}**: {result}")
        except Exception as e:
            st.error(f"❌ Error in '{name}': {e}")

    # ======================
    # 5. Full EDA Report
    # ======================
    st.header("📑 Full Report")
    if st.button("📊 Generate Full EDA Report"):
        with st.spinner("Generating..."):
            profile = ProfileReport(data, explorative=True)
            st_profile_report(profile)
