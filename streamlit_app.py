import streamlit as st
import pandas as pd
from data_utils import load_data, plot_mood_trend, plot_burnout_distribution, plot_average_by_phase, plot_phase_distribution
from cycle_tracking import calculate_menstrual_phase # Add this line
from datetime import date, timedelta # Add this line as we need date objects
import os # Import os for file operations

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


# --- Configuration (Redefine here for clarity and consistency with previous responses) ---
DATA_FILE = 'cleaned_data.csv' # Keeping this here as it's used by save_data and load_data

# --- Helper Functions (From your original structure, with `load_data` columns updated) ---
@st.cache_data
def load_data():
    """Loads existing data or creates an empty DataFrame if file doesn't exist."""
    try:
        df = pd.read_csv(DATA_FILE)
        # Ensure 'date' column is datetime type for potential sorting/analysis
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.info("No existing data found. A new CSV file will be created upon first submission.")
        # Define columns based on the full survey structure
        # ENSURE THESE COLUMNS MATCH THE KEYS IN new_record WHEN SAVING
        return pd.DataFrame(columns=[
            'date', 'age', 'mood', 'anxiety', 'energy', 'burnout', 'journal_entry',
            'sleep', 'refreshed_after_sleep',
            'steps_taken', 'water_intake', 'caffeine_intake',
            'work_hours', 'outfit_type',
            'music_genre', 'music_volume', 'music_time',
            'on_period_today', 'cycle_phase', 'symptoms_experienced',
            'major_event_log'
        ])

def save_data(df):
    """Saves the DataFrame to the CSV file."""
    df.to_csv(DATA_FILE, index=False)


# --- Mapping Dictionaries (Consolidated and ensured consistency with your survey) ---
burnout_map = {
    'Not at all': 0,
    'Mild tiredness': 1,
    'Drained': 2,
    'Full-on burnout': 3,
}

refreshed_map_input = { # For mapping user input to numerical for CSV
    'Yes': 2,
    'Kind of': 1,
    'No': 0
}

steps_map_input = {
    '<2000': 0,
    '2000–5000': 1, # Use en dash consistently as per your earlier code
    '5000–8000': 2,
    '8000+': 3,
}

water_map_input = {
    '<1L': 0,
    '1–1.5L': 1, # Use en dash
    '1.5–2.5L': 2,
    '2.5L+': 3
}

caffeine_map_input = {
    'None': 0,
    '1 cup': 1,
    '2 cups': 2,
    '3+ cups': 3
}

work_hours_map_input = { # Renamed from work_map for clarity with input
    '<2 hrs': 0,
    '2–5 hrs': 1, # Use en dash
    '5–8 hrs': 2,
    '8+ hrs': 3,
}

outfit_map_input = {
    'Comfy (PJs, oversized)': 0,
    'Casual (jeans, tee)': 1,
    'Formal (workwear)': 2,
    'Athletic/Activewear': 3,
}

music_volume_map_input = { # Renamed from volume_map for clarity with input
    'Low (background)': 0,
    'Medium': 1,
    'Loud': 2,
}

# --- Load data for use in pages (Using your existing df variable) ---
df = load_data()
# Your original code converts date format here, let's keep it for compatibility
# But ideally, load_data should handle it robustly or the CSV should be consistent.
# For now, let's just make sure it's datetime
df['date'] = pd.to_datetime(df['date'])


# Pages (Your original structure, untouched)
if page == "🏠 Home":
    st.title("Welcome to Burnout Tracker 🧠")
    st.markdown("Track your mood, health, and lifestyle to understand burnout patterns.")
    st.metric("Data entries", len(df))
    st.metric("Average Mood", round(df['mood'].mean(), 1))
    # Note: df['burnout'] might contain string values, adjust for this if you want to count
    # 'full-on burnout' from the loaded data.
    # We will ensure the saving uses the mapped integer value in Daily Log.
    st.metric("Burnout Entries (Full-on)", df['burnout'].astype(str).str.lower().value_counts().get('full-on burnout', 0))

elif page == "📊 Visualize Trends":
    st.title("📈 Mood & Burnout Trends")
    plot_mood_trend(df)
    plot_burnout_distribution(df)
    st.markdown("---") # Add a separator for clarity
    st.subheader("Trends by Menstrual Phase 🌸")

    # Call the new plotting functions
    plot_average_by_phase(df, 'mood', 'Average Mood by Menstrual Phase')
    # Ensure 'burnout' is numeric or handle it in plot_average_by_phase if it takes strings
    # For now, if 'burnout' column in CSV is string, plot_average_by_phase might need adaptation.
    # Assuming the data is cleaned to numerical for plots.
    plot_average_by_phase(df, 'burnout', 'Average Burnout by Menstrual Phase')
    plot_average_by_phase(df, 'energy', 'Average Energy by Menstrual Phase')
    plot_phase_distribution(df) # Shows distribution of your data points across phases

elif page == "📝 Log Your Day":
    st.title("📝 Log Your Day")
    st.write("Enter your daily metrics to track your journey.")

    # Load existing data for display after submission
    existing_df_for_display = load_data()

    # --- Daily Log Form (Modified to include all survey features) ---
    with st.form("daily_log_form", clear_on_submit=True): # Wrapped in st.form for clarity and reset
        st.header("🗓️ Today's Log")

        # Meta Information
        log_date = st.date_input("What is today's date?", date.today(), help="Defaults to today, adjust if logging for a different day.")
        # Age should ideally be a static user profile input, but if it's logged daily:
        age = st.number_input("How old are you?", min_value=10, max_value=100, value=30, help="Your current age.")

        st.subheader("🧠 Core Wellbeing")
        mood = st.slider("How would you rate your mood today? (1-10, 10 being best)", 1, 10, 5)
        anxiety = st.slider("How anxious did you feel today? (1-10, 10 being worst)", 1, 10, 5)
        energy = st.slider("What was your energy level like today? (1-10, 10 being most energetic)", 1, 10, 5)
        burnout_options = ["Not at all", "Mild tiredness", "Drained", "Full-on burnout"]
        burnout = st.radio("How burned out do you feel today?", burnout_options, index=0)
        journal_entry = st.text_area("Anything you'd like to note about today?", "", help="Free text input for your thoughts or feelings.")

        st.subheader("😴 Sleep & Recovery")
        sleep_hours = st.number_input("How many hours did you sleep?", min_value=0.0, max_value=16.0, value=7.0, step=0.5)
        refreshed_after_sleep_options = ['Yes', 'Kind of', 'No'] # Renamed from refreshed_options
        refreshed_after_sleep = st.radio("Did you feel refreshed after sleeping?", refreshed_after_sleep_options, index=1)

        st.subheader("🚶‍♀️ Activity & Hydration")
        steps_options = ['<2000', '2000–5000', '5000–8000', '8000+'] # Corrected to en-dash for consistency
        steps_taken = st.radio("How many steps did you take today?", steps_options, index=1)

        water_options = ['<1L', '1–1.5L', '1.5–2.5L', '2.5L+'] # Corrected to en-dash for consistency
        water_intake = st.radio("How much water did you drink today?", water_options, index=1)

        caffeine_options = ['None', '1 cup', '2 cups', '3+ cups']
        caffeine_intake = st.radio("How much caffeine did you consume?", caffeine_options, index=0)

        st.subheader("💻 Work & Routine")
        work_hours_options = ['<2 hrs', '2–5 hrs', '5–8 hrs', '8+ hrs'] # Corrected to en-dash
        work_hours_val = st.radio("How many hours did you work or study today?", work_hours_options, index=1) # Renamed to avoid conflict

        outfit_options = ['Comfy (PJs, oversized)', 'Casual (jeans, tee)', 'Formal (workwear)', 'Athletic/Activewear']
        outfit_type = st.radio("What kind of outfit were you in today?", outfit_options, index=1)

        st.subheader("🎧 Music Habits")
        music_genre_options = ["None", "Carnatic", "Western classical", "Lofi", "Sad Pop", "Rock", "EDM", "Happy Vibey pop", "Dance music", "Bollywood", "K-pop / J-pop", "Other"]
        music_genre_selected = st.multiselect("What music did you listen to today?", music_genre_options, default=["None"])
        music_genre_str = ", ".join(music_genre_selected) if music_genre_selected else "None"

        music_volume_options = ['Low (background)', 'Medium', 'Loud']
        music_volume = st.radio("How loud was the music?", music_volume_options, index=1)

        music_time_options = ['Morning', 'Afternoon', 'Night']
        music_time = st.radio("When did you listen to music?", music_time_options, index=0)

        st.subheader("🩸 Cycle Tracking")
        on_period_options = ['Yes', 'No', 'Not sure']
        on_period_today = st.radio("Were you on your period today?", on_period_options, index=1)

        # Conditional display for Cycle Phase and Symptoms based on daily log input
        # Note: This is separate from the sidebar Smart Cycle Tracker
        daily_log_cycle_phase = "N/A - Not tracked"
        daily_log_symptoms_experienced_str = "None"

        if on_period_today == 'Yes' or on_period_today == 'Not sure':
            cycle_phase_options = ['PMS', 'Follicular', 'Ovulation', 'Luteal', 'No clue']
            daily_log_cycle_phase = st.radio("Which phase of your cycle are you in (for this log)?", cycle_phase_options, index=4)

            symptoms_options = ['Cramps', 'Low energy', 'Food craving', 'None']
            daily_log_symptoms_selected = st.multiselect("Any symptoms experienced today?", symptoms_options, default=["None"])
            daily_log_symptoms_experienced_str = ", ".join(daily_log_symptoms_selected) if daily_log_symptoms_selected else "None"


        major_event_log = st.text_area("Major event log (e.g., travel, new project)", "")


        submitted = st.form_submit_button("Save Daily Log")

        if submitted:
            # Calculate menstrual phase for the logged date using user's cycle info from sidebar
            # This is specifically for the 'cycle_phase' column in the CSV,
            # which could be separate from the daily log input for consistency.
            # However, your prompt implies the daily log phase should be used,
            # so we'll use daily_log_cycle_phase for the CSV column.
            
            # Prepare new data record, ensuring column names match 'load_data' and survey
            new_record = {
                'date': log_date.strftime('%Y-%m-%d'),
                'age': age,
                'mood': mood,
                'anxiety': anxiety,
                'energy': energy,
                'burnout': burnout, # Store as string, can be mapped to int during analysis
                'journal_entry': journal_entry,
                'sleep': sleep_hours,
                'refreshed_after_sleep': refreshed_map_input[refreshed_after_sleep], # Map to numerical
                'steps_taken': steps_map_input[steps_taken], # Map to numerical
                'water_intake': water_map_input[water_intake], # Map to numerical
                'caffeine_intake': caffeine_map_input[caffeine_intake], # Map to numerical
                'work_hours': work_hours_map_input[work_hours_val], # Map to numerical
                'outfit_type': outfit_map_input[outfit_type], # Map to numerical
                'music_genre': music_genre_str, # Store as comma-separated string
                'music_volume': music_volume_map_input[music_volume], # Map to numerical
                'music_time': music_time,
                'on_period_today': on_period_today, # Store as 'Yes', 'No', 'Not sure'
                'cycle_phase': daily_log_cycle_phase, # Use the daily log input for phase
                'symptoms_experienced': daily_log_symptoms_experienced_str, # Store as comma-separated string
                'major_event_log': major_event_log
            }

            new_df = pd.DataFrame([new_record])

            # Append new record to existing data
            updated_df = pd.concat([existing_df_for_display, new_df], ignore_index=True)

            # Remove potential duplicates for a given date, keeping the last entry
            updated_df = updated_df.drop_duplicates(subset=['date'], keep='last')

            # Sort by date for chronological order
            updated_df['date'] = pd.to_datetime(updated_df['date'])
            updated_df = updated_df.sort_values(by='date').reset_index(drop=True)

            # Save updated data
            save_data(updated_df)
            st.success("Your daily log has been saved!")
            st.experimental_rerun() # Rerun to show updated data or clear form

    # --- Display Current Data (Moved outside form for persistent display) ---
    st.subheader("Your Recent Logs")
    if not existing_df_for_display.empty:
        # Ensure 'date' is datetime before formatting or sorting
        existing_df_for_display['date'] = pd.to_datetime(existing_df_for_display['date'])
        # Display the latest 10 entries for quick review
        st.dataframe(existing_df_for_display.tail(10).style.format({'sleep': '{:.1f}'}))
    else:
        st.info("No logs yet. Fill out the form above to start tracking!")