import streamlit as st
import pandas as pd
# Assuming data_utils.py and cycle_tracking.py are in the same directory
from data_utils import load_data, plot_mood_trend, plot_burnout_distribution, plot_average_by_phase, plot_phase_distribution
from cycle_tracking import calculate_menstrual_phase
from datetime import date, timedelta
import os
import json
import numpy as np

# Import the prediction function. It now handles model loading internally.
from predict import predict_burnout
# This utility function from your data processing script is still needed for advice logic
from burnout_data_processor import convert_intake_to_numeric_value

# --- IMPORTANT: st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Burnout Tracker", layout="centered")
# --- END IMPORTANT ---


# --- Global Configurations ---
user_id = 'real_user_000' # This user_id must match the one used in your training data/script

# --- Load external data (symptoms, music genres) ---
def load_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: {file_path} not found. Please ensure it's in the same directory.")
        return []
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return []

all_symptoms = load_json_data('all_symptoms.json')
all_music_genres = load_json_data('all_music_genres.json')

# --- Data File Paths ---
DAILY_LOG_DATA_FILE = 'cleaned_data.csv'
USER_PROFILE_DATA_FILE = 'user_profile.csv'

# The model file path is defined in predict.py, but it's good to keep this variable here
# if you need to reference the file existence as you do below.
STANDARD_MODEL_FILE = 'burnout_prediction_random_forest_model_tuned_with_bert_nlp_and_domain_features_v2.pkl'


# --- Load and Save Data Utilities ---
def save_daily_log_data(df):
    df.to_csv(DAILY_LOG_DATA_FILE, index=False)

def load_daily_log_data():
    return load_data(DAILY_LOG_DATA_FILE)

def save_user_profile_data(df):
    df.to_csv(USER_PROFILE_DATA_FILE, index=False)

def load_user_profile_data():
    return load_data(USER_PROFILE_DATA_FILE)

# Load existing data at the start
existing_daily_logs_df = load_daily_log_data()
existing_user_profile_df = load_user_profile_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📝 Log Your Day", "📊 Visualize Trends", "👤 User Profile"])

st.sidebar.markdown("---")
st.sidebar.title("🌸 Smart Cycle Tracker")

if "cycle_start_date" not in st.session_state:
    st.session_state.cycle_start_date = date.today() - timedelta(days=10)
if "cycle_length" not in st.session_state:
    st.session_state.cycle_length = 28

st.session_state.cycle_start_date = st.sidebar.date_input(
    "When did your last period start?",
    value=st.session_state.cycle_start_date,
    key="cycle_start_date_input"
)
st.session_state.cycle_length = st.sidebar.number_input(
    "Average cycle length (days):",
    min_value=20,
    max_value=45,
    value=st.session_state.cycle_length,
    key="cycle_length_input"
)

if st.session_state.cycle_start_date and st.session_state.cycle_length:
    today_date_for_cycle = date.today()
    try:
        cycle_info = calculate_menstrual_phase(
            current_date=today_date_for_cycle,
            cycle_start_date=st.session_state.cycle_start_date,
            cycle_length=st.session_state.cycle_length
        )
        st.sidebar.write(f"**Today's Cycle Phase:** {cycle_info['phase']} (Day {cycle_info['cycle_day']})")
        st.session_state.current_cycle_phase = cycle_info['phase']
        st.session_state.on_period_today = (cycle_info['phase'] == 'Period')
    except ValueError as e:
        st.sidebar.error(f"Cycle calculation error: {e}")
        st.session_state.current_cycle_phase = "Unknown"
        st.session_state.on_period_today = False
else:
    st.sidebar.write("Please enter cycle details to track your phase.")
    st.session_state.current_cycle_phase = "Unknown"
    st.session_state.on_period_today = False


# --- Home Page ---
if page == "🏠 Home":
    st.title("Welcome to Your Burnout Tracker 🚀")
    st.write("""
    This application helps you track daily metrics, visualize trends, and even predict burnout risk
    based on your inputs and menstrual cycle data.
    """)
    st.markdown("---")
    st.subheader("How It Works:")
    st.markdown("""
    * **📝 Log Your Day:** Enter your daily mood, energy, sleep, journal entries, and more.
    * **📊 Visualize Trends:** See how your mood, energy, and anxiety change across different cycle phases and over time.
    * **🌸 Smart Cycle Tracker:** Input your last period start date and average cycle length in the sidebar to automatically track your current menstrual phase.
    * **👤 User Profile:** Set up or update your personal profile information.
    * **🔮 Burnout Prediction:** Get a daily prediction of your current burnout level right here on the home page!
    """)
    st.markdown("---")
    st.subheader("Your Latest Burnout Prediction & Advice")

    latest_daily_log = pd.DataFrame()
    if not existing_daily_logs_df.empty and 'user_id' in existing_daily_logs_df.columns:
        user_daily_logs = existing_daily_logs_df[existing_daily_logs_df['user_id'] == user_id].copy()
        if not user_daily_logs.empty:
            user_daily_logs['date'] = pd.to_datetime(user_daily_logs['date'], errors='coerce')
            latest_daily_log = user_daily_logs.sort_values(by='date', ascending=True).iloc[[-1]]

    latest_user_profile = pd.DataFrame()
    if not existing_user_profile_df.empty and 'user_id' in existing_user_profile_df.columns:
        user_profiles = existing_user_profile_df[existing_user_profile_df['user_id'] == user_id].copy()
        if not user_profiles.empty:
            latest_user_profile = user_profiles.iloc[[-1]]

    if latest_daily_log.empty or latest_user_profile.empty:
        st.info("Please go to '📝 Log Your Day' to log today's data and '👤 User Profile' to set up your profile for a burnout prediction.")
    else:
        # Check for the existence of the trained model file
        # This check is still valid to inform the user if the model is missing
        if not os.path.exists(STANDARD_MODEL_FILE):
            st.warning(f"Model file ('{STANDARD_MODEL_FILE}') not found.")
            st.info(f"Please ensure you have run `python train_burnout_model.py` in your terminal to train the model for user '{user_id}'.")
            st.info(f"Make sure the generated model file is in the same directory as this app.")
            st.stop()

        try:
            prediction_result_home = predict_burnout(
                current_day_raw_log=latest_daily_log.copy(),
                latest_user_profile=latest_user_profile.copy(),
                historical_daily_logs=existing_daily_logs_df.copy()
            )

            if prediction_result_home and "Error" not in prediction_result_home.get('predicted_label', ''):
                st.info(f"Based on your latest log ({latest_daily_log['date'].iloc[0].strftime('%Y-%m-%d')}) and profile:")
                st.markdown(f"### Predicted Burnout Level: **{prediction_result_home['predicted_label']}**")

                st.write("Confidence:")
                prob_data = prediction_result_home['probabilities']
                if prob_data:
                    # Ensure keys are correct for sorting and display
                    # If label_mapping provides string labels, sort by original class (0, 1, 2, 3)
                    # Convert keys to int for sorting if they are numerical class labels
                    prob_data_cleaned = {}
                    for k, v in prob_data.items():
                        # Try to convert key to int if it looks like a class label, otherwise keep as string
                        try:
                            prob_data_cleaned[int(k)] = v
                        except ValueError:
                            prob_data_cleaned[k] = v

                    if all(isinstance(k, int) for k in prob_data_cleaned.keys()):
                         sorted_prob_items = sorted(prob_data_cleaned.items(), key=lambda item: item[0])
                         prob_df_home = pd.DataFrame(sorted_prob_items, columns=['Burnout Level', 'Probability'])
                    else: # If keys are already string labels, sort alphabetically or keep original order
                         prob_df_home = pd.DataFrame(list(prob_data_cleaned.items()), columns=['Burnout Label', 'Probability'])


                    prob_df_home['Probability'] = prob_df_home['Probability'].map('{:.2%}'.format)
                    st.dataframe(prob_df_home, hide_index=True)
                else:
                    st.write("Probabilities not available.")

                st.markdown("---")
                st.subheader("Personalized Advice for You 💡")
                advice_messages = []

                # Ensure 'sleep' column is numeric for min() calculation
                recent_sleep_data = existing_daily_logs_df[existing_daily_logs_df['user_id'] == user_id].copy()
                if 'sleep' in recent_sleep_data.columns:
                    recent_sleep_data['sleep'] = pd.to_numeric(recent_sleep_data['sleep'], errors='coerce')
                    recent_sleep_data = recent_sleep_data.sort_values(by='date', ascending=False).head(3)['sleep']
                else:
                    recent_sleep_data = pd.Series() # Empty series if column is missing

                if not recent_sleep_data.empty and not recent_sleep_data.isnull().all():
                    min_sleep_past_3_days = recent_sleep_data.min()
                    recommended_sleep = 8.0
                    if min_sleep_past_3_days < recommended_sleep:
                        advice_messages.append(f"- Your sleep over the last three days (min: **{min_sleep_past_3_days:.1f} hours**) is below the recommended 8 hours. Try to get at least **8 hours of quality sleep**.")
                else:
                    advice_messages.append("- Log more daily data to get personalized sleep advice.")

                if 'water_intake' in latest_daily_log.columns and not pd.isna(latest_daily_log['water_intake'].iloc[0]):
                    water_val = latest_daily_log['water_intake'].iloc[0]
                    latest_water_intake_numeric = convert_intake_to_numeric_value(water_val, 'water_intake')

                    if not pd.isna(latest_water_intake_numeric) and latest_water_intake_numeric < 2.0:
                        advice_messages.append(f"- You logged **{water_val}** of water. Aim for at least **2 liters of water** daily to stay hydrated.")
                else:
                    advice_messages.append("- Log your water intake to get personalized hydration advice.")

                advice_messages.append("- Consider incorporating **yoga and meditation** into your routine to reduce stress.")

                alcohol_val = latest_user_profile.get('alcohol_consumption', None)
                # Ensure alcohol_val is extracted correctly if it's a Series from .get()
                if isinstance(alcohol_val, pd.Series):
                    alcohol_val = alcohol_val.iloc[0] if not alcohol_val.empty else None

                if alcohol_val is not None and not pd.isna(alcohol_val):
                    if "More than 2 drinks/day" in str(alcohol_val):
                        advice_messages.append("- Your alcohol consumption is noted as 'More than 2 drinks/day'. Consider **reducing alcohol intake** for better well-being.")
                    elif "1-2 drinks/day" in str(alcohol_val):
                        advice_messages.append("- Your alcohol consumption is noted as '1-2 drinks/day'. Be mindful of your intake and consider moderation for optimal health.")

                drug_val = latest_user_profile.get('drug_use', None)
                if isinstance(drug_val, pd.Series):
                    drug_val = drug_val.iloc[0] if not drug_val.empty else None

                if drug_val is not None and not pd.isna(drug_val):
                    if str(drug_val) == "Yes":
                        advice_messages.append("- Your drug_use is noted as 'Yes'. Consider **reducing or stopping drug use** for your health and well-being.")

                if advice_messages:
                    for advice in advice_messages:
                        st.markdown(advice)
                else:
                    st.success("Keep up the great work! Your recent logs indicate a healthy routine.")

            else:
                st.error(f"Could not get a prediction: {prediction_result_home.get('predicted_label', 'Unknown error')}")
                st.info("Ensure you have logged daily data and set up your user profile, and the model is correctly loaded.")
        except Exception as e:
            st.error(f"An error occurred while predicting burnout for your latest log: {e}")
            st.info("Please ensure your model file is in the correct directory and data is available.")


# --- Log Your Day Page ---
elif page == "📝 Log Your Day":
    st.title("📝 Log Your Day")
    st.write("Capture your daily experiences to build a comprehensive record.")

    today = date.today()

    with st.form(key='daily_log_form'):
        st.subheader("Daily Metrics")
        log_date = st.date_input("Date", value=today)
        mood = st.slider("Mood (1-10, 1=Awful, 10=Fantastic)", 1, 10, 5)
        sleep = st.number_input("Hours of Sleep", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        anxiety = st.slider("Anxiety (1-10, 1=None, 10=Severe)", 1, 10, 5)
        energy = st.slider("Energy (1-10, 1=Exhausted, 10=Energized)", 1, 10, 5)
        refreshed_after_sleep = st.selectbox("Refreshed after sleep?", ["Yes", "No"])

        water_intake_options = ["0-0.5L", "0.5-1.5L", "1.5-2.5L", "2.5-3.5L", "3.5L+"]
        water_intake = st.selectbox("Daily Water Intake", water_intake_options)

        caffeine_intake_options = ["None", "1 cup", "2 cups", "3 cups", "4+ cups"]
        caffeine_intake = st.selectbox("Daily Caffeine Intake", caffeine_intake_options)

        work_hours = st.number_input("Hours Worked", min_value=0.0, value=8.0, step=0.5)
        steps_taken = st.number_input("Steps Taken", min_value=0, value=5000, step=500)
        outfit_type = st.selectbox("Outfit Type", ["Casual", "Smart Casual", "Formal", "Sporty", "Loungewear"])

        music_preferences_expanded = st.checkbox("Track Music Preferences?")
        music_genre_input = "Not Tracked"
        music_volume_input = "Not Tracked"
        music_time_input = 0.0 # Changed default to 0.0 for numeric consistency

        if music_preferences_expanded:
            selected_music_genres = st.multiselect("Music Genre(s) Listened To", all_music_genres)
            music_genre_input = ", ".join(selected_music_genres) if selected_music_genres else "None"
            music_volume_input = st.selectbox("Music Volume", ["Low", "Medium", "High", "Not Applicable"])
            music_time_input = st.number_input("Hours Listening to Music", min_value=0.0, value=0.0, step=0.5)

        # Ensure cycle phase and on_period are correctly set based on sidebar and override
        on_period_today_input = st.checkbox("On Period Today (Override Cycle Tracker)", value=st.session_state.on_period_today)
        cycle_phase_input = st.session_state.current_cycle_phase
        if on_period_today_input:
            cycle_phase_input = "Period" # If overridden, explicitly set to Period

        symptoms_experienced = st.multiselect("Symptoms Experienced", all_symptoms)
        symptoms_experienced_str = ", ".join(symptoms_experienced) if symptoms_experienced else "None"

        journal_entry = st.text_area("Journal Entry (optional)", "Write about your day, feelings, and thoughts...")
        major_event_log = st.text_area("Major Event Log (optional)", "Any significant events today?")

        submitted = st.form_submit_button("Save Daily Log")

        if submitted:
            new_record = {
                'user_id': user_id,
                'date': log_date.isoformat(),
                'mood': mood,
                'sleep': sleep,
                'anxiety': anxiety,
                'energy': energy,
                'refreshed_after_sleep': refreshed_after_sleep,
                'water_intake': water_intake,
                'caffeine_intake': caffeine_intake,
                'work_hours': work_hours,
                'steps_taken': steps_taken,
                'outfit_type': outfit_type,
                'music_genre': music_genre_input,
                'music_volume': music_volume_input,
                'music_time': music_time_input, # This should be a number, ensure consistency
                'on_period_today': "Yes" if on_period_today_input else "No", # Store as string "Yes"/"No"
                'cycle_phase': cycle_phase_input,
                'symptoms_experienced': symptoms_experienced_str,
                'major_event_log': major_event_log,
                'journal_entry': journal_entry,
                'burnout': np.nan # Default to NaN for new logs
            }

            new_df = pd.DataFrame([new_record])
            updated_df = pd.concat([existing_daily_logs_df, new_df], ignore_index=True)
            updated_df = updated_df.drop_duplicates(subset=['user_id', 'date'], keep='last')
            updated_df['date'] = pd.to_datetime(updated_df['date'])
            updated_df = updated_df.sort_values(by=['user_id', 'date']).reset_index(drop=True)
            save_daily_log_data(updated_df)
            st.success("Your daily log has been saved!")
            st.rerun()

    st.subheader("Your Recent Logs")
    if not existing_daily_logs_df.empty and 'user_id' in existing_daily_logs_df.columns:
        user_daily_logs_display = existing_daily_logs_df[existing_daily_logs_df['user_id'] == user_id].copy()
        if not user_daily_logs_display.empty:
            user_daily_logs_display['date'] = pd.to_datetime(user_daily_logs_display['date'])
            st.dataframe(user_daily_logs_display.sort_values(by='date', ascending=False).head(10), hide_index=True)
        else:
            st.info(f"No logs yet for `user_id={user_id}`. Start by filling out the form above!")
    else:
        st.info("No logs yet or `user_id` column is missing. Start by filling out the form above!")


# --- Visualize Trends Page ---
elif page == "📊 Visualize Trends":
    st.title("📊 Your Burnout & Well-being Trends")
    st.write("Explore how your daily metrics change over time and across your menstrual cycle phases.")

    if existing_daily_logs_df.empty:
        st.warning("No data available to display trends. Please log some days first on the 'Log Your Day' page.")
    else:
        # Ensure 'date' is datetime before sorting
        existing_daily_logs_df['date'] = pd.to_datetime(existing_daily_logs_df['date'])
        existing_daily_logs_df = existing_daily_logs_df.sort_values(by='date').reset_index(drop=True)

        st.subheader("Mood, Energy, and Anxiety Over Time")
        # Ensure columns are numeric for plotting
        for col in ['mood', 'energy', 'anxiety']:
            if col in existing_daily_logs_df.columns:
                existing_daily_logs_df[col] = pd.to_numeric(existing_daily_logs_df[col], errors='coerce')

        if 'mood' in existing_daily_logs_df.columns and 'energy' in existing_daily_logs_df.columns and 'anxiety' in existing_daily_logs_df.columns:
            st.plotly_chart(plot_mood_trend(existing_daily_logs_df, 'mood', 'Mood Over Time'), use_container_width=True)
            st.plotly_chart(plot_mood_trend(existing_daily_logs_df, 'energy', 'Energy Over Time'), use_container_width=True)
            st.plotly_chart(plot_mood_trend(existing_daily_logs_df, 'anxiety', 'Anxiety Over Time'), use_container_width=True)
        else:
            st.info("Insufficient 'mood', 'energy', or 'anxiety' data for trend plots. Log more daily entries.")

        st.subheader("Burnout Level Distribution")
        if 'burnout' in existing_daily_logs_df.columns and not existing_daily_logs_df['burnout'].isnull().all():
            st.plotly_chart(plot_burnout_distribution(existing_daily_logs_df), use_container_width=True)
        else:
            st.info("No sufficient 'burnout' data available to display distribution. Log more daily entries with burnout levels (or let the prediction run).")

        st.subheader("Average Metrics by Cycle Phase")
        if 'cycle_phase' in existing_daily_logs_df.columns and not existing_daily_logs_df['cycle_phase'].isnull().all():
            for col_metric in ['mood', 'energy', 'anxiety']:
                if col_metric in existing_daily_logs_df.columns:
                    st.plotly_chart(plot_average_by_phase(existing_daily_logs_df, col_metric, f'Average {col_metric.capitalize()} by Cycle Phase'), use_container_width=True)
                else:
                    st.info(f"Insufficient '{col_metric}' data for 'Average {col_metric.capitalize()} by Cycle Phase' plot.")
        else:
            st.info("No sufficient 'cycle_phase' data for phase-based average plots. Ensure cycle tracker is configured and log more entries.")

        st.subheader("Cycle Phase Distribution")
        if 'cycle_phase' in existing_daily_logs_df.columns and not existing_daily_logs_df['cycle_phase'].isnull().all():
            st.plotly_chart(plot_phase_distribution(existing_daily_logs_df), use_container_width=True)
        else:
            st.info("No sufficient 'cycle_phase' data for 'Cycle Phase Distribution' plot. Ensure cycle tracker is configured and log more entries.")


# --- User Profile Page ---
elif page == "👤 User Profile":
    st.title("👤 Your User Profile")
    st.write("Set up or update your personal information. This data helps in personalized predictions.")

    profile_defaults = {
        'age': 25, 'occupation': 'Student', 'alcohol_consumption': 'None', 'drug_use': 'No',
        'smoking_habits': 'No', 'contraceptive_pill_use': 'No', 'prior_history_burnout': 'No',
        'prior_history_anxiety': 'No', 'living_situation': 'Alone', 'supportive_environment': 'Yes',
        'liking_study_work_environment': 'Yes'
    }

    current_profile = {}
    if not existing_user_profile_df.empty and 'user_id' in existing_user_profile_df.columns:
        user_profile_data = existing_user_profile_df[existing_user_profile_df['user_id'] == user_id]
        if not user_profile_data.empty:
            current_profile = user_profile_data.iloc[-1].to_dict()

    with st.form(key='user_profile_form'):
        age = st.number_input("Age", min_value=18, max_value=100, value=int(current_profile.get('age', profile_defaults['age'])))
        occupation = st.text_input("Occupation", value=current_profile.get('occupation', profile_defaults['occupation']))

        alc_options = ["None", "1-2 drinks/day (occasional/moderate)", "More than 2 drinks/day (regular/heavy)"]
        current_alc_value = current_profile.get('alcohol_consumption', profile_defaults['alcohol_consumption'])
        # Handle cases where current_alc_value might be NaN or not in options
        if pd.isna(current_alc_value) or current_alc_value not in alc_options:
            selected_alc_index = alc_options.index(profile_defaults['alcohol_consumption']) # Use default if current is invalid
        else:
            selected_alc_index = alc_options.index(current_alc_value)
        alcohol_consumption = st.selectbox("Alcohol Consumption", alc_options, index=selected_alc_index)

        drug_options = ["Yes", "No", "Prefer not to say"]
        current_drug_value = current_profile.get('drug_use', profile_defaults['drug_use'])
        if pd.isna(current_drug_value) or current_drug_value not in drug_options:
            selected_drug_index = drug_options.index(profile_defaults['drug_use'])
        else:
            selected_drug_index = drug_options.index(current_drug_value)
        drug_use = st.selectbox("Drug Use", drug_options, index=selected_drug_index)

        smoking_options = ["Yes", "No", "Prefer not to say"]
        current_smoking_value = current_profile.get('smoking_habits', profile_defaults['smoking_habits'])
        if pd.isna(current_smoking_value) or current_smoking_value not in smoking_options:
            selected_smoking_index = smoking_options.index(profile_defaults['smoking_habits'])
        else:
            selected_smoking_index = smoking_options.index(current_smoking_value)
        smoking_habits = st.selectbox("Smoking Habits", smoking_options, index=selected_smoking_index)

        contraceptive_options = ["Yes", "No", "Prefer not to say"]
        current_contraceptive_value = current_profile.get('contraceptive_pill_use', profile_defaults['contraceptive_pill_use'])
        if pd.isna(current_contraceptive_value) or current_contraceptive_value not in contraceptive_options:
            selected_contraceptive_index = contraceptive_options.index(profile_defaults['contraceptive_pill_use'])
        else:
            selected_contraceptive_index = contraceptive_options.index(current_contraceptive_value)
        contraceptive_pill_use = st.selectbox("Contraceptive Pill Use", contraceptive_options, index=selected_contraceptive_index)

        burnout_history_options = ["Yes", "No", "Prefer not to say"]
        current_burnout_history_value = current_profile.get('prior_history_burnout', profile_defaults['prior_history_burnout'])
        if pd.isna(current_burnout_history_value) or current_burnout_history_value not in burnout_history_options:
            selected_burnout_history_index = burnout_history_options.index(profile_defaults['prior_history_burnout'])
        else:
            selected_burnout_history_index = burnout_history_options.index(current_burnout_history_value)
        prior_history_burnout = st.selectbox("Prior History of Burnout", burnout_history_options, index=selected_burnout_history_index)

        anxiety_history_options = ["Yes", "No", "Prefer not to say"]
        current_anxiety_history_value = current_profile.get('prior_history_anxiety', profile_defaults['prior_history_anxiety'])
        if pd.isna(current_anxiety_history_value) or current_anxiety_history_value not in anxiety_history_options:
            selected_anxiety_history_index = anxiety_history_options.index(profile_defaults['prior_history_anxiety'])
        else:
            selected_anxiety_history_index = anxiety_history_options.index(current_anxiety_history_value)
        prior_history_anxiety = st.selectbox("Prior History of Anxiety", anxiety_history_options, index=selected_anxiety_history_index)

        living_situation = st.text_input("Living Situation", value=current_profile.get('living_situation', profile_defaults['living_situation']))

        support_env_options = ["Yes", "No"]
        current_support_env_value = current_profile.get('supportive_environment', profile_defaults['supportive_environment'])
        if pd.isna(current_support_env_value) or current_support_env_value not in support_env_options:
            selected_support_env_index = support_env_options.index(profile_defaults['supportive_environment'])
        else:
            selected_support_env_index = support_env_options.index(current_support_env_value)
        supportive_environment = st.selectbox("Supportive Home/Work Environment", support_env_options, index=selected_support_env_index)

        liking_env_options = ["Yes", "No"]
        current_liking_env_value = current_profile.get('liking_study_work_environment', profile_defaults['liking_study_work_environment'])
        if pd.isna(current_liking_env_value) or current_liking_env_value not in liking_env_options:
            selected_liking_env_index = liking_env_options.index(profile_defaults['liking_study_work_environment'])
        else:
            selected_liking_env_index = liking_env_options.index(current_liking_env_value)
        liking_study_work_environment = st.selectbox("Liking Your Study/Work Environment", liking_env_options, index=selected_liking_env_index)

        profile_submitted = st.form_submit_button("Save Profile")

        if profile_submitted:
            new_profile_record = {
                'user_id': user_id,
                'age': age,
                'occupation': occupation,
                'alcohol_consumption': alcohol_consumption,
                'drug_use': drug_use,
                'smoking_habits': smoking_habits,
                'contraceptive_pill_use': contraceptive_pill_use,
                'prior_history_burnout': prior_history_burnout,
                'prior_history_anxiety': prior_history_anxiety,
                'living_situation': living_situation,
                'supportive_environment': supportive_environment,
                'liking_study_work_environment': liking_study_work_environment
            }
            new_profile_df = pd.DataFrame([new_profile_record])

            if not existing_user_profile_df.empty and 'user_id' in existing_user_profile_df.columns:
                updated_profile_df = pd.concat([existing_user_profile_df, new_profile_df], ignore_index=True)
                updated_profile_df = updated_profile_df.drop_duplicates(subset=['user_id'], keep='last')
            else:
                updated_profile_df = new_profile_df

            save_user_profile_data(updated_profile_df)
            st.success("Your user profile has been saved!")
            st.rerun()

    st.subheader("Current Profile")
    if not existing_user_profile_df.empty and 'user_id' in existing_user_profile_df.columns:
        user_current_profile_display = existing_user_profile_df[existing_user_profile_df['user_id'] == user_id].copy()
        if not user_current_profile_display.empty:
            st.dataframe(user_current_profile_display, hide_index=True)
        else:
            st.info(f"No profile data yet for `user_id={user_id}`. Fill out the form above!")
    else:
        st.info("No profile data file found or `user_id` column is missing. Fill out the form above to create your profile!")