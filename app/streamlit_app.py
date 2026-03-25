import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Email Classifier", layout="wide")

st.title("📩 AI Powered Smart Email Classifier")
st.markdown("Classifies enterprise emails into categories and detects urgency levels.")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state['history'] = []

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI Prediction URL", "http://localhost:8000/predict")
    
tab1, tab2 = st.tabs(["Classify Email", "Dashboard"])

with tab1:
    email_text = st.text_area("Enter Email Content", height=200, placeholder="Paste email here...")
    
    if st.button("Classify Email", type="primary"):
        if not email_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(api_url, json={"email": email_text})
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state['history'].append({
                            "email": email_text,
                            "category": data["category"],
                            "urgency": data["urgency"]
                        })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Category", data["category"])
                        with col2:
                            urgency = data["urgency"]
                            color = "red" if urgency == "High" else "orange" if urgency == "Medium" else "green"
                            st.markdown(f"### Urgency: <span style='color:{color}'>{urgency}</span>", unsafe_allow_html=True)
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to API. Is the FastAPI server running?")

with tab2:
    st.header("Analysis Dashboard")
    history_df = pd.DataFrame(st.session_state['history'])
    
    if history_df.empty:
        st.info("No emails classified yet. Predict some emails to populate the dashboard.")
    else:
        # Filters
        st.subheader("Filter Predictions")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sel_cat = st.multiselect("Filter by Category", options=history_df['category'].unique(), default=history_df['category'].unique())
        with col_f2:
            sel_urg = st.multiselect("Filter by Urgency", options=history_df['urgency'].unique(), default=history_df['urgency'].unique())
            
        filtered_df = history_df[(history_df['category'].isin(sel_cat)) & (history_df['urgency'].isin(sel_urg))]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Charts
        st.subheader("Visualizations")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.write("Distribution by Category")
            fig1, ax1 = plt.subplots(figsize=(5,3))
            sns.countplot(data=filtered_df, x='category', ax=ax1, palette='viridis', hue='category', legend=False)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)
            
        with col_c2:
            st.write("Urgency Trends")
            fig2, ax2 = plt.subplots(figsize=(5,3))
            palette_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            sns.countplot(data=filtered_df, x='urgency', ax=ax2, palette=palette_map, hue='urgency', legend=False)
            plt.tight_layout()
            st.pyplot(fig2)
