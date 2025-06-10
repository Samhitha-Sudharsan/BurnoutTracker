import pandas as pd
from datetime import date
from cycle_tracking import calculate_menstrual_phase

# Load CSV
df = pd.read_csv("data.csv")
df['date'] = df['date'].astype(str).str.replace('–', '-')
# Convert 'date' column to datetime objects first, then to date objects
# Specify dayfirst=True to handle DD-MM-YYYY format
df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.date

# --- Menstrual Cycle Tracking Integration Setup ---
# For batch processing of historical data in data_cleaning.py,
# we'll use a placeholder/average cycle start and length.
# IMPORTANT: Adjust this `sample_cycle_start_date` to reflect the general period start
# date relevant to your `data.csv` entries, so phases make sense.
# For example, if your data.csv entries are from 2024, set this to a date in 2024.
# If your data is from multiple years, pick a consistent start date for the earliest cycle in your data.
sample_cycle_start_date = date(2024, 1, 15) # <<< CRITICAL: Adjust this date based on your 'data.csv' earliest entries
sample_cycle_length = 28 # Average cycle length

# Rename columns for consistency
# This line should come after initial data loading and basic date conversion,
# but before specific column processing.
df.columns = ['date', 'age', 'major_event_log', 'mood', 'anxiety', 'energy', 'burnout', 'journal_entry', 'sleep',
              'refreshed', 'steps', 'water', 'caffeine', 'work', 'outfit', 'music', 'volume', 'music_time',
              'period', 'phase', 'symptom']

# Convert numerical columns
numerical_cols = ['age', 'mood', 'anxiety', 'energy', 'sleep']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Function to clean strings: replace en-dash with hyphen, lowercase, strip spaces
def clean_string_col(col):
    df[col] = df[col].fillna('none').astype(str)
    df[col] = df[col].str.replace('–', '-')  # en-dash to hyphen
    df[col] = df[col].str.lower().str.strip()
    df[col] = df[col].replace({'nan': 'none'})
    
# Clean relevant categorical columns before mapping
categorical_cols_to_clean = ['burnout', 'refreshed', 'steps', 'water', 'caffeine', 'work', 'volume', 'outfit']
for col in categorical_cols_to_clean:
    clean_string_col(col)

# Categorical to numeric mappings
burnout_map = {
    'not at all': 0,
    'mild tiredness': 1,
    'drained': 2,
    'full-on burnout': 3,
    'none': None
}

refreshed_map = {
    'yes': 2,
    'kind of': 1,
    'no': 0,
    'none': None
}

steps_map = {
    '<2000': 0,
    '2000-5000': 1,
    '5000-8000': 2,
    '8000+': 3,
    'none': None
}

water_map = {
    '<1l': 0,
    '1-1.5l': 1,
    '1.5-2.5l': 2,
    '2.5l+': 3,
    'none': None
}

caffeine_map = {
    'none': 0,
    '1 cup': 1,
    '2 cups': 2,
    '3+ cups': 3
}

work_map = {
    '<2 hrs': 0,
    '2-5 hrs': 1,
    '5-8 hrs': 2,
    '8+ hrs': 3,
    'none': None
}

volume_map = {
    'low (background)': 0,
    'medium': 1,
    'loud': 2,
    'none': None
}

outfit_map = {
    'comfy (pjs, oversized)': 0,
    'casual (jeans, tee)': 1,
    'formal (workwear)': 2,
    'athletic/activewear': 3,
    'none': None
}

# Apply mappings
df['burnout'] = df['burnout'].map(burnout_map)
df['refreshed'] = df['refreshed'].map(refreshed_map)
df['steps'] = df['steps'].map(steps_map)
df['water'] = df['water'].map(water_map)
df['caffeine'] = df['caffeine'].map(caffeine_map)
df['work'] = df['work'].map(work_map)
df['volume'] = df['volume'].map(volume_map)
df['outfit'] = df['outfit'].map(outfit_map)

# Calculate menstrual phase for each entry
phases = []
cycle_days = []
for index, row in df.iterrows():
    current_date_entry = row['date']
    
    # Ensure cycle_start_date is before or on current_date_entry
    # If the data entry is before our fixed sample_cycle_start_date,
    # we'll handle it as 'N/A'. You should ideally set sample_cycle_start_date
    # to be before or around the start of your actual data.
    if current_date_entry < sample_cycle_start_date:
        phase_info = {"phase": "N/A", "cycle_day": None}
    else:
        phase_info = calculate_menstrual_phase(
            current_date_entry,
            sample_cycle_start_date,
            sample_cycle_length
        )
    phases.append(phase_info['phase'])         # <<< THIS IS NOW CORRECTLY INDENTED
    cycle_days.append(phase_info['cycle_day']) # <<< THIS IS NOW CORRECTLY INDENTED

df['menstrual_phase'] = phases
df['cycle_day_of_phase'] = cycle_days # Renamed to avoid confusion with overall cycle day

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("✅ Data cleaned and menstrual phases added to cleaned_data.csv")