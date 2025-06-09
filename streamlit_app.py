# streamlit_app.py
import streamlit as st
import pandas as pd
from data_utils import load_data, plot_mood_trend, plot_burnout_distribution

st.set_page_config(page_title="Burnout Tracker", layout="centered")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualize Trends", "📝 Log Your Day"])

# Load data
df = load_data()

# Pages
if page == "🏠 Home":
    st.title("Welcome to Burnout Tracker 🧠")
    st.markdown("Track your mood, health, and lifestyle to understand burnout patterns.")
    st.metric("Data entries", len(df))
    st.metric("Average Mood", round(df['mood'].mean(), 1))
    st.metric("Burnout Entries", df['burnout'].value_counts().get("full-on burnout", 0))

elif page == "📊 Visualize Trends":
    st.title("📈 Mood & Burnout Trends")
    plot_mood_trend(df)
    plot_burnout_distribution(df)

elif page == "📝 Log Your Day":
    st.title("📝 Daily Entry")
    st.markdown("This form is for demo. Real data comes from your Google Form CSV.")
    mood = st.slider("Mood (1–10)", 1, 10, 5)
    energy = st.slider("Energy (1–10)", 1, 10, 5)
    burnout = st.selectbox("Burnout level", ["not at all", "mild tiredness", "drained", "full-on burnout"])
    st.button("Submit (demo only)")

