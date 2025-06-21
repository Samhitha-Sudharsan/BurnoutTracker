import pandas as pd
import numpy as np
from datetime import date, timedelta
import random

# --- Configuration for Synthetic Training Data Generation ---
TRAINING_DATA_FILE = 'model_training_data.csv' # Output file for model training data
num_synthetic_users = 100    # How many different "people" to simulate for training
days_per_user = 90           # How many days of data per simulated person
total_rows_to_generate = num_synthetic_users * days_per_user

print(f"Generating synthetic training data for {num_synthetic_users} simulated users, {days_per_user} days each, totaling {total_rows_to_generate} rows...")

# Define ranges/options for features - BASED ON YOUR SURVEY
age_range = (18, 65) # Wider age range for training diversity

# Core Wellbeing
mood_range = (1, 10)
anxiety_range = (1, 10)
energy_range = (1, 10)
burnout_options_list = ["Not at all", "Mild tiredness", "Drained", "Full-on burnout"]

# Sleep & Recovery
sleep_hours_range = (0.0, 12.0)
refreshed_options = ['Yes', 'No', 'Kind of']

# Activity & Hydration
steps_options = ['<2000', '2000–5000', '5000–8000', '8000+']
water_options = ['<1L', '1–1.5L', '1.5–2.5L', '2.5L+']
caffeine_options = ['None', '1 cup', '2 cups', '3+ cups']

# Work & Routine
work_hours_options = ['<2 hrs', '2–5 hrs', '5–8 hrs', '8+ hrs']
outfit_options = ['Comfy (PJs, oversized)', 'Casual (jeans, tee)', 'Formal (workwear)', 'Athletic/Activewear']

# Music Habits
music_genre_options = ["None", "Carnatic", "Western classical", "Lofi", "Sad Pop", "Rock", "EDM", "Happy Vibey pop", "Dance music", "Bollywood", "K-pop / J-pop", "Other"]
music_volume_options = ['Low (background)', 'Medium', 'Loud']
music_time_options = ['Morning', 'Afternoon', 'Night']

# Cycle Tracking
on_period_options = ['Yes', 'No', 'Not sure']
# Simplified cycle phase logic for diverse training data
cycle_phase_synthetic_map = {
    'Menstrual': (1, 5),
    'Follicular': (6, 14),
    'Ovulation': (15, 16),
    'Luteal': (17, 28)
}
symptoms_options = ['Cramps', 'Low energy', 'Food craving', 'None']

all_records = []

# --- Generate Data for Each Synthetic User ---
for user_idx in range(num_synthetic_users):
    user_id = f"user_{user_idx:03d}"
    user_age = random.randint(age_range[0], age_range[1])

    # Simulate if this user tracks their cycle and their specific cycle parameters
    # Approx 50% of simulated users track cycles, 50% don't
    tracks_cycle = random.random() < 0.5
    menstrual_cycle_length = random.randint(25, 35) if tracks_cycle else 0
    
    # Simulate a realistic cycle start date for this user relative to their data range
    # Data will run from today's date backwards
    user_data_end_date = date.today()
    user_data_start_date = user_data_end_date - timedelta(days=days_per_user - 1)
    
    cycle_start_date_for_user = None
    if tracks_cycle:
        # Ensure cycle start date is within a reasonable window for the generated data
        # E.g., start cycle in the last 2*cycle_length days relative to data end
        cycle_start_date_for_user = user_data_end_date - timedelta(days=random.randint(0, menstrual_cycle_length * 2))

    # Generate data for 'days_per_user' for each user
    for day_offset in range(days_per_user):
        log_date = user_data_start_date + timedelta(days=day_offset)

        record = {
            'date': log_date.strftime('%Y-%m-%d'),
            'user_id': user_id, # Identifier for simulated user
            'age': user_age
        }

        # --- Core Wellbeing ---
        record['mood'] = random.randint(mood_range[0], mood_range[1])
        record['anxiety'] = random.randint(anxiety_range[0], anxiety_range[1])
        record['energy'] = random.randint(energy_range[0], energy_range[1])
        record['burnout'] = random.choice(burnout_options_list)
        record['journal_entry'] = random.choice(["", "Feeling well today.", "A bit tired.", "Productive day at work.", "Relaxed evening."]) if random.random() < 0.3 else ""

        # --- Sleep & Recovery ---
        record['sleep'] = round(random.uniform(sleep_hours_range[0], sleep_hours_range[1]), 1)
        record['refreshed_after_sleep'] = random.choice(refreshed_options)

        # --- Activity & Hydration ---
        record['steps_taken'] = random.choice(steps_options)
        record['water_intake'] = random.choice(water_options)
        record['caffeine_intake'] = random.choice(caffeine_options)

        # --- Work & Routine ---
        record['work_hours'] = random.choice(work_hours_options)
        record['outfit_type'] = random.choice(outfit_options)

        # --- Music Habits ---
        num_genres = random.randint(0, 2)
        if num_genres == 0 or "None" in random.sample([g for g in music_genre_options], 1): # Small chance of "None"
            record['music_genre'] = 'None'
        else:
            selected_genres = random.sample([g for g in music_genre_options if g != 'None'], min(num_genres, len(music_genre_options)-1))
            record['music_genre'] = ', '.join(selected_genres)

        record['music_volume'] = random.choice(music_volume_options)
        record['music_time'] = random.choice(music_time_options)

        # --- Cycle Tracking ---
        if tracks_cycle and cycle_start_date_for_user:
            days_since_cycle_start = (log_date - cycle_start_date_for_user).days
            if days_since_cycle_start >= 0:
                cycle_day_in_period = (days_since_cycle_start % menstrual_cycle_length) + 1

                # On Period Today?
                record['on_period_today'] = 'Yes' if 1 <= cycle_day_in_period <= 7 else 'No'
                if random.random() < 0.05: record['on_period_today'] = 'Not sure' # Small chance of 'Not sure'

                # Cycle Phase
                if 1 <= cycle_day_in_period <= cycle_phase_synthetic_map['Menstrual'][1]:
                    record['cycle_phase'] = 'Menstrual'
                elif cycle_phase_synthetic_map['Follicular'][0] <= cycle_day_in_period <= cycle_phase_synthetic_map['Follicular'][1]:
                    record['cycle_phase'] = 'Follicular'
                elif cycle_phase_synthetic_map['Ovulation'][0] <= cycle_day_in_period <= cycle_phase_synthetic_map['Ovulation'][1]:
                    record['cycle_phase'] = 'Ovulation'
                else:
                    record['cycle_phase'] = 'Luteal'
                if random.random() < 0.03: record['cycle_phase'] = 'No clue' # Small chance of 'No clue'

                # Symptoms Experienced
                sim_symptoms = []
                if record['cycle_phase'] == 'Menstrual' and random.random() < 0.8:
                    sim_symptoms.append(random.choice(['Cramps', 'Low energy']))
                    if random.random() < 0.4: sim_symptoms.append('Food craving')
                elif record['cycle_phase'] == 'Luteal' and random.random() < 0.5:
                    sim_symptoms.append(random.choice(['Low energy', 'Food craving']))
                
                if not sim_symptoms:
                    record['symptoms_experienced'] = 'None'
                else:
                    record['symptoms_experienced'] = ', '.join(sim_symptoms)
            else: # If log_date is before the synthetic cycle start date for this user
                record['on_period_today'] = 'No'
                record['cycle_phase'] = 'No clue'
                record['symptoms_experienced'] = 'None'
        else: # For users who don't track cycle (or if 'log_date' is too far out)
            record['on_period_today'] = 'No'
            record['cycle_phase'] = 'No clue'
            record['symptoms_experienced'] = 'None'

        record['major_event_log'] = random.choice([
            "None", "Travelled for work", "Started new hobby", "Felt sick", "Attended a social event"
        ]) if random.random() < 0.1 else ""

        all_records.append(record)

# Create DataFrame
synthetic_df = pd.DataFrame(all_records)

# Ensure 'date' is datetime for sorting
synthetic_df['date'] = pd.to_datetime(synthetic_df['date'])

# Sort data by user and then by date, helpful for time-series analysis per user
synthetic_df = synthetic_df.sort_values(by=['user_id', 'date']).reset_index(drop=True)

# Final check of columns to match your survey's structure for training data
# Note: 'major_event_log' was added to the end of your Streamlit app's data collection,
# so ensure it's here too for consistency with the columns used in model training.
expected_training_columns = [
    'date', 'user_id', 'age', 'mood', 'anxiety', 'energy', 'burnout', 'journal_entry',
    'sleep', 'refreshed_after_sleep',
    'steps_taken', 'water_intake', 'caffeine_intake',
    'work_hours', 'outfit_type',
    'music_genre', 'music_volume', 'music_time',
    'on_period_today', 'cycle_phase', 'symptoms_experienced',
    'major_event_log'
]
# Reorder columns to match the desired order, handling potential missing/extra columns
final_columns = [col for col in expected_training_columns if col in synthetic_df.columns]
for col in synthetic_df.columns:
    if col not in final_columns:
        final_columns.append(col) # Add any extra generated columns at the end if needed

synthetic_df = synthetic_df[final_columns]

# Save to CSV
synthetic_df.to_csv(TRAINING_DATA_FILE, index=False)

print(f"\nSynthetic training data generated and saved to {TRAINING_DATA_FILE}")
print(f"Total rows generated: {len(synthetic_df)}")
print("\nFirst 5 rows of synthetic training data:")
print(synthetic_df.head())
print("\nAge distribution (sample of first 10 unique ages):")
print(synthetic_df['age'].value_counts().sort_index().head(10))
print("\nBurnout level distribution in synthetic training data:")
print(synthetic_df['burnout'].value_counts(normalize=True))
print("\nSample of On Period Today values:")
print(synthetic_df['on_period_today'].value_counts(normalize=True))
print("\nSample of Cycle Phases:")
print(synthetic_df['cycle_phase'].value_counts(normalize=True))