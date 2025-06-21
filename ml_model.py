# import pandas as pd
# import numpy as np # Import numpy for NaN handling

# # Load the cleaned data
# try:
#     df = pd.read_csv("cleaned_data.csv")
#     print("Data loaded successfully from cleaned_data.csv")
#     print(f"Initial DataFrame shape: {df.shape}")
# except FileNotFoundError:
#     print("Error: cleaned_data.csv not found. Please ensure data_cleaning.py has been run.")
#     exit()

# # --- Initial Data Inspection ---

# print("\n--- DataFrame Info ---")
# df.info()

# print("\n--- First 5 rows ---")
# print(df.head())

# print("\n--- Descriptive Statistics (Numerical Columns) ---")
# print(df.describe())

# print("\n--- Missing Values Check ---")
# print(df.isnull().sum())

# print("\n--- Unique values and counts for key categorical/numerical columns ---")
# print("\nBurnout distribution:")
# print(df['burnout'].value_counts(dropna=False)) # Include NaN counts

# print("\nMenstrual Phase distribution:")
# print(df['menstrual_phase'].value_counts(dropna=False))

# print("\nMood distribution:")
# print(df['mood'].value_counts(dropna=False).sort_index())

# print("\nEnergy distribution:")
# print(df['energy'].value_counts(dropna=False).sort_index())

# print("\nRefreshed distribution:")
# print(df['refreshed'].value_counts(dropna=False))

# # Ensure 'date' is a datetime object for time-based analysis if needed later
# df['date'] = pd.to_datetime(df['date'])
# print("\n'date' column type after conversion:", df['date'].dtype)

# print("\n--- Initial Data Exploration Complete ---")
# print("Review the output above to understand your dataset's characteristics.")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # New import for splitting data
from sklearn.preprocessing import OneHotEncoder # New import for encoding categorical features
from sklearn.impute import SimpleImputer # New import for imputation
from sklearn.ensemble import RandomForestClassifier # New import for the model
from sklearn.metrics import classification_report, accuracy_score, f1_score # New imports for evaluation

# --- 1. Load Data ---
try:
    df = pd.read_csv("cleaned_data.csv")
    print("Data loaded successfully from cleaned_data.csv")
    print(f"Initial DataFrame shape: {df.shape}")
except FileNotFoundError:
    print("Error: cleaned_data.csv not found. Please ensure data_cleaning.py has been run.")
    exit()

# Ensure 'date' is a datetime object for time-based analysis if needed later
df['date'] = pd.to_datetime(df['date'])



# --- 2. Data Preprocessing for ML ---

print("\n--- Starting Data Preprocessing for ML ---")

# Sort data by date to enable correct time-series feature creation
df = df.sort_values(by='date').reset_index(drop=True)
print("Data sorted by date for time-series feature creation.")

# Drop columns that won't be used for the first ML model (text fields, redundant 'phase')
columns_to_drop = ['major_event_log', 'journal_entry', 'symptom', 'phase']
df = df.drop(columns=columns_to_drop, errors='ignore') # 'errors=ignore' prevents error if column not found
print(f"Dropped columns: {columns_to_drop}")

# Handle missing values in the target variable ('burnout')
initial_rows = df.shape[0]
df.dropna(subset=['burnout'], inplace=True)
print(f"Dropped {initial_rows - df.shape[0]} row(s) with missing 'burnout' values.")

# Impute missing numerical/categorical-mapped values with the mode
numerical_features_to_impute = ['outfit', 'volume', 'cycle_day_of_phase']
numerical_imputer = SimpleImputer(strategy='most_frequent')
df[numerical_features_to_impute] = numerical_imputer.fit_transform(df[numerical_features_to_impute])
print(f"Imputed missing values in {numerical_features_to_impute} with mode.")

# Impute missing 'menstrual_phase' with a new category 'Unknown'
df['menstrual_phase'].fillna('Unknown', inplace=True)
print("Imputed missing 'menstrual_phase' with 'Unknown'.")

# --- Feature Engineering (Adding temporal and contextual features) ---
print("\n--- Starting Feature Engineering ---")

# Convert relevant columns to numeric before calculating rolling averages/lags, handling potential errors
# This is crucial as original 'mood', 'anxiety', 'energy' might be int, but we want float for rolling means
for col in ['mood', 'anxiety', 'energy', 'sleep', 'steps', 'water', 'caffeine', 'work']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Rolling Averages (e.g., last 3 days, 7 days)
for window in [3, 7]:
    for col in ['mood', 'anxiety', 'energy', 'sleep', 'steps', 'water', 'work']:
        df[f'{col}_avg_{window}d'] = df.groupby('age')[col].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        # Use groupby('age') to ensure rolling calculations are per person, not across all dates
        # fillna with the individual's mean for early entries where window isn't full
        df[f'{col}_avg_{window}d'].fillna(df[col], inplace=True) # Fill NaNs for rolling mean with current day's value if window not met

print(f"Added rolling averages for mood, anxiety, energy, sleep, steps, water, work over 3 and 7 days.")

# Lagged Features (e.g., value from the previous day)
for lag in [1]: # Can add more lags if needed, e.g., 2, 3
    for col in ['mood', 'anxiety', 'energy', 'sleep', 'refreshed', 'steps', 'water', 'caffeine', 'work']:
        df[f'{col}_lag_{lag}d'] = df.groupby('age')[col].shift(lag)
        # For lagged features, fill NaNs (for first 'lag' days of each person) with the mean of that column for that person
        df[f'{col}_lag_{lag}d'].fillna(df[col].mean(), inplace=True) # Or fill with 0 or the current day's value. Mean is a common robust choice.

print(f"Added lagged features for mood, anxiety, energy, sleep, refreshed, steps, water, caffeine, work (1 day lag).")

# Day of Week (0=Monday, 6=Sunday)
df['day_of_week'] = df['date'].dt.dayofweek
print("Added 'day_of_week' feature.")

print("\n--- Feature Engineering Complete ---")


# --- Feature Encoding (One-Hot Encoding for categorical variables) ---
# Identify columns that need one-hot encoding
# Include 'day_of_week' as a new categorical feature
features_to_encode = ['period', 'music', 'music_time', 'menstrual_phase', 'day_of_week']

# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the categorical features
encoded_features = encoder.fit_transform(df[features_to_encode])

# Create a DataFrame from the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(features_to_encode), index=df.index)

# Drop original categorical columns and concatenate encoded ones
df = df.drop(columns=features_to_encode)
df = pd.concat([df, encoded_df], axis=1)
print(f"Applied One-Hot Encoding to: {features_to_encode}")

# --- Define Features (X) and Target (Y) ---
# Ensure 'date' is dropped from X as it's not a direct feature for the model
X = df.drop(columns=['burnout', 'date'])
Y = df['burnout']

print(f"\nFeatures (X) shape after preprocessing: {X.shape}")
print(f"Target (Y) shape after preprocessing: {Y.shape}")
print("\nFirst 5 rows of X (Features) after Engineering and Encoding:")
print(X.head())
print("\nFirst 5 rows of Y (Target):")
print(Y.head())

print("\n--- Data Preprocessing Complete ---")
print("Your data is now ready for model training!")



# --- 3. Split Data into Training and Testing Sets ---
# We'll use a test size of 20% (0.2) of your data, and a random_state for reproducibility
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print(f"\n--- Data Splitting Complete ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")

# --- 4. Model Training ---
print(f"\n--- Starting Model Training (RandomForestClassifier) ---")

# Initialize the Random Forest Classifier
# random_state for reproducibility, class_weight='balanced' for potential imbalance in target classes
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train, Y_train)

print(f"Model training complete.")

# --- 5. Model Evaluation ---
print(f"\n--- Evaluating Model Performance ---")

# Make predictions on the test set
Y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")
# f1 = f1_score(Y_test,Y_pred)
# print(f"f1 score on test set: {f1:.2f}")

# Print classification report (provides precision, recall, f1-score for each class)
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("\n--- Model Training and Evaluation Complete ---")

# You can save this preprocessed data if you want to inspect it
# df.to_csv("preprocessed_data_for_ml.csv", index=False)