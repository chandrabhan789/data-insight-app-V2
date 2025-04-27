import streamlit as st
import pandas as pd
import json
from pathlib import Path
import io

# ======================
# 1. Setup: Load/Save Insights (No Dependencies)
# ======================
INSIGHTS_FILE = "saved_insights.json"

def load_insights():
    """Load insights without external dependencies"""
    try:
        if Path(INSIGHTS_FILE).exists():
            with open(INSIGHTS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_insights(insights):
    """Save insights without external dependencies"""
    try:
        with open(INSIGHTS_FILE, "w") as f:
            json.dump(insights, f)
        return True
    except Exception:
        return False

# ======================
# 2. Streamlit App (Pure Insights)
# ======================
st.set_page_config(page_title="Insights AI", layout="wide")
st.title("🔍 Pure Insights Generator")
st.markdown("Upload/paste data → Teach AI → Get insights")

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
            st.error(f"❌ Error: {str(e)}")
else:
    uploaded_file = st.file_uploader("Upload Data (CSV/Excel/JSON)", type=["csv", "xlsx", "json"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.success("✅ Data loaded!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if data is not None:
    # ======================
    # 3. Auto-Basic Analysis (No Graphs)
    # ======================
    st.header("📊 Basic Analysis")
    
    # Numeric Summary
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        st.subheader("Numeric Columns")
        st.dataframe(data[numeric_cols].describe().round(2))
    
    # Text Summary
    text_cols = data.select_dtypes(include='object').columns.tolist()
    if text_cols:
        st.subheader("Text Columns")
        for col in text_cols:
            st.write(f"**{col}**: {data[col].nunique()} unique values")

    # ======================
    # 4. Trainable Insights (Core Feature)
    # ======================
    st.header("🎓 Teach AI Insights")
    insights = load_insights()

    # Input New Insight
    col1, col2 = st.columns(2)
    with col1:
        insight_name = st.text_input("Insight Name:")
    with col2:
        insight_logic = st.text_area("Logic (Use `data` in Python):", height=100,
                                   placeholder="data['Sales'].mean() > 100")

    if st.button("💾 Save Insight") and insight_name and insight_logic:
        try:
            # Test the logic
            test_result = eval(insight_logic, {'data': data.head(1)})
            insights[insight_name] = insight_logic
            if save_insights(insights):
                st.success(f"✅ Saved: '{insight_name}'!")
            else:
                st.error("❌ Failed to save (check permissions)")
        except Exception as e:
            st.error(f"❌ Invalid logic: {str(e)}")

    # Show Saved Insights
    st.subheader("🔍 Your Insights")
    for name, logic in insights.items():
        try:
            result = eval(logic, {'data': data})
            st.write(f"✅ **{name}**: {result}")
        except Exception as e:
            st.error(f"❌ Error in '{name}': {str(e)}")

    # ======================
    # 5. Data Export (Optional)
    # ======================
    st.header("📤 Export")
    if st.button("📝 Show Full Data"):
        st.dataframe(data)
