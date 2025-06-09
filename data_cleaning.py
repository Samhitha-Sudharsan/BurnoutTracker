import pandas as pd

# Load CSV
df = pd.read_csv("data.csv")

# Rename columns for consistency
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

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("✅ Data cleaned and saved as 'cleaned_data.csv'")
