# streamlit_app.py
import streamlit as st
import pandas as pd
from data_utils import load_data, plot_mood_trend, plot_burnout_distribution, plot_average_by_phase, plot_phase_distribution
from cycle_tracking import calculate_menstrual_phase # Add this line 
from datetime import date, timedelta # Add this line as we need date objects


st.set_page_config(page_title="Burnout Tracker", layout="centered")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualize Trends", "📝 Log Your Day"])

st.sidebar.markdown("---") # Separator
st.sidebar.title("🌸 Smart Cycle Tracker") # New section title

    # Input fields for cycle data
    # Use st.session_state to persist these inputs across reruns
if "cycle_start_date" not in st.session_state:
        st.session_state.cycle_start_date = date.today() - timedelta(days=10) # Default to 10 days ago
if "cycle_length" not in st.session_state:
        st.session_state.cycle_length = 28 # Default cycle length

st.session_state.cycle_start_date = st.sidebar.date_input(
        "When did your last period start?",
        value=st.session_state.cycle_start_date,
        key="cycle_start_date_input"
    )
st.session_state.cycle_length = st.sidebar.number_input(
        "Approx. cycle length (days)?",
        min_value=20,
        max_value=45,
        value=st.session_state.cycle_length,
        key="cycle_length_input"
    )

    # Calculate and display current phase
current_date_for_phase = date.today()
    # Ensure we only calculate if the start date is valid (not in the future)
if st.session_state.cycle_start_date and st.session_state.cycle_start_date <= current_date_for_phase:
        current_phase_info = calculate_menstrual_phase(
            current_date_for_phase,
            st.session_state.cycle_start_date,
            st.session_state.cycle_length
        )
        st.sidebar.subheader("Current Cycle Phase:")
        st.sidebar.markdown(f"**Phase:** {current_phase_info['phase']}")
        if current_phase_info['cycle_day']:
            st.sidebar.markdown(f"**Cycle Day:** {current_phase_info['cycle_day']}")
else:
        st.sidebar.info("Enter your last period start date to see your current cycle phase.")
# Load data

df = load_data()
# df['date'] = df['date'].str.replace('–', '-', regex=False)  # Replace en dash with hyphen
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")


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
    st.markdown("---") # Add a separator for clarity
    st.subheader("Trends by Menstrual Phase 🌸")

    # Call the new plotting functions
    plot_average_by_phase(df, 'mood', 'Average Mood by Menstrual Phase')
    plot_average_by_phase(df, 'burnout', 'Average Burnout by Menstrual Phase')
    plot_average_by_phase(df, 'energy', 'Average Energy by Menstrual Phase')
    plot_phase_distribution(df) # Shows distribution of your data points across phases
elif page == "📝 Log Your Day":
    st.title("📝 Daily Entry")
    st.markdown("This form is for demo. Real data comes from your Google Form CSV.")
    mood = st.slider("Mood (1–10)", 1, 10, 5)
    energy = st.slider("Energy (1–10)", 1, 10, 5)
    burnout = st.selectbox("Burnout level", ["not at all", "mild tiredness", "drained", "full-on burnout"])
    st.button("Submit (demo only)")

