import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Save Insights Functionality
INSIGHTS_FILE = "saved_insights.json"

def load_insights():
    if Path(INSIGHTS_FILE).exists():
        with open(INSIGHTS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_insights(insights):
    with open(INSIGHTS_FILE, "w") as f:
        json.dump(insights, f)

# Streamlit App
st.set_page_config(page_title="AI Insights Generator", layout="wide")
st.title("AI Data Insights Generator")
st.markdown("Upload your data and get high-level insights!")

# Data Input Method: Paste or Upload
input_method = st.radio("Input Method", ("📋 Paste Data", "📂 Upload Data"))
data = None

if input_method == "📋 Paste Data":
    raw_data = st.text_area("Paste CSV/JSON Data", height=300)
    if st.button("Parse Data"):
        try:
            if raw_data.strip()[0] in ['{', '[']:  # It's JSON
                data = pd.read_json(StringIO(raw_data))
            else:  # It's CSV
                data = pd.read_csv(StringIO(raw_data))
            st.success("Data parsed successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

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
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

# Display the first few rows of the dataset
if data is not None:
    st.dataframe(data.head())

    # Data Type Identification
    st.subheader("Identified Data Types")
    st.write(data.dtypes)

    # Analyzing Numeric Columns
    st.subheader("Summary Statistics for Numeric Data")
    numeric_data = data.select_dtypes(include=np.number)
    if not numeric_data.empty:
        st.write(numeric_data.describe())

    # Trend Identification: Correlation Analysis
    st.subheader("Correlation Heatmap")
    if not numeric_data.empty:
        corr = numeric_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        st.pyplot()

    # Logical Analysis: Checking Missing Data
    st.subheader("Missing Data")
    missing_data = data.isnull().sum()
    st.write(missing_data)

    # Generate High-Level Insights
    st.subheader("High-Level Insights")
    insights = []
    if not numeric_data.empty:
        for col in numeric_data.columns:
            insights.append(f"Average of {col}: {numeric_data[col].mean():.2f}")
            insights.append(f"Max of {col}: {numeric_data[col].max()}")
            insights.append(f"Min of {col}: {numeric_data[col].min()}")

    if not insights:
        insights.append("No numeric data available for insights.")
    
    for insight in insights[:10]:
        st.write(insight)

    # Save Insights (Optional)
    st.subheader("Save Custom Insight")
    custom_insight = st.text_input("Enter a custom insight name")
    insight_logic = st.text_area("Enter insight logic (Python expression using 'data')")

    if st.button("Save Insight") and custom_insight and insight_logic:
        try:
            # Test if the logic works
            result = eval(insight_logic, {"data": data})
            custom_insights = load_insights()
            custom_insights[custom_insight] = insight_logic
            save_insights(custom_insights)
            st.success(f"Insight '{custom_insight}' saved!")
        except Exception as e:
            st.error(f"Error in logic: {e}")

    # Applying Saved Insights
    st.subheader("Applied Custom Insights")
    custom_insights = load_insights()
    for name, logic in custom_insights.items():
        try:
            result = eval(logic, {"data": data})
            st.write(f"Insight: {name} - Result: {result}")
        except Exception as e:
            st.error(f"Error in custom insight '{name}': {e}")
