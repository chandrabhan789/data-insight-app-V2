
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import io

# ======================
# 1. Setup: Load/Save Insights
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
# 2. Streamlit App (Insights Only)
# ======================
st.set_page_config(page_title="AI Insights Generator", layout="wide")
st.title("🔍 Pure Insights AI")
st.markdown("Upload/paste data → Teach AI → Get insights!")

# ----- Data Input -----
input_method = st.radio("Input Method:", ("📋 Paste Data", "📂 Upload File"))
data = None

if input_method == "📋 Paste Data":
    raw_text = st.text_area("Paste CSV/JSON:", height=200,
                          placeholder="CSV Example:\nDay,Sales\nMon,100\nTue,150")
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
    # 3. Auto-Generated Basic Insights
    # ======================
    st.header("📊 Automatic Findings")
    
    # Numeric Columns Summary
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Numeric Columns")
        st.write(data[numeric_cols].describe())
    
    # Text Columns Summary
    text_cols = data.select_dtypes(include='object').columns.tolist()
    if text_cols:
        st.subheader("Text Columns")
        st.write("Unique values per column:")
        for col in text_cols:
            st.write(f"- {col}: {data[col].nunique()} unique values")

    # ======================
    # 4. Trainable Insights System
    # ======================
    st.header("🎓 Teach AI Custom Insights")
    insights = load_insights()

    # A. Add New Insight
    new_insight_name = st.text_input("Insight Name (e.g., 'Weekend Drop'):")
    new_insight_logic = st.text_area("Logic (Python using `data`):",
                                   placeholder="data[data['Day'].isin(['Sat','Sun'])]['Sales'].mean()")
    
    if st.button("💾 Save Insight") and new_insight_name and new_insight_logic:
        try:
            # Test if the logic works
            test_result = eval(new_insight_logic, {'data': data.head()})
            insights[new_insight_name] = new_insight_logic
            save_insights(insights)
            st.success(f"✅ Saved: '{new_insight_name}'!")
        except Exception as e:
            st.error(f"❌ Invalid logic: {e}")

    # B. Apply Learned Insights
    st.subheader("🔎 Your Custom Insights")
    for name, logic in insights.items():
        try:
            result = eval(logic, {'data': data})
            st.write(f"✅ **{name}**: {result}")
        except Exception as e:
            st.error(f"❌ Error in '{name}': {e}")

    # ======================
    # 5. One-Click Full Report
    # ======================
    st.header("📑 Full Data Report")
    if st.button("📊 Generate Full Analysis"):
        with st.spinner("Creating report..."):
            profile = ProfileReport(data, explorative=True)
            st_profile_report(profile)
