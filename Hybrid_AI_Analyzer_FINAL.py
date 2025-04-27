import streamlit as st
import pandas as pd
import numpy as np
import io

# Streamlit app title
st.title("AI Insights Generator - Data Analysis")

# 1. Data Upload or Paste Option
input_method = st.radio("Select Input Method", ("📋 Paste Data", "📂 Upload File"))
data = None

# If user selects Paste Data
if input_method == "📋 Paste Data":
    raw_text = st.text_area("Paste CSV/JSON Data:", height=200, placeholder="CSV Example:\nDay,Sales\nMon,100\nTue,150")
    if st.button("Parse Data") and raw_text:
        try:
            # Attempt to parse CSV or JSON
            data = pd.read_json(io.StringIO(raw_text)) if raw_text.strip()[0] in ['{', '['] else pd.read_csv(io.StringIO(raw_text))
            st.success("✅ Data parsed successfully!")
        except Exception as e:
            st.error(f"❌ Error parsing data: {e}")

# If user selects Upload File
else:
    uploaded_file = st.file_uploader("Upload Data (CSV/Excel/JSON)", type=["csv", "xlsx", "json"])
    if uploaded_file:
        try:
            # Check file extension and load the data
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                data = pd.read_json(uploaded_file)
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")

# 2. Data Type Identification
if data is not None:
    st.subheader("📊 Data Overview")
    st.write(data.head())  # Show first few rows of the data

    st.subheader("📋 Data Types")
    # Identify and display the data types of each column
    data_types = data.dtypes
    st.write(data_types)

    # Detect numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("📈 Numeric Columns Summary")
        st.write(data[numeric_cols].describe())
    else:
        st.write("No numeric columns found.")

    # Detect categorical (text) columns
    text_cols = data.select_dtypes(include='object').columns.tolist()
    if text_cols:
        st.subheader("🔤 Text Columns Summary")
        st.write("Unique values per column:")
        for col in text_cols:
            st.write(f"- {col}: {data[col].nunique()} unique values")
    else:
        st.write("No text columns found.")
