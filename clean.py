import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 1. Load Data ---
# Ensure 'final_combined_training_data.csv' is in the same directory as this script,
# or provide the full path to the file.
print("Loading data from final_combined_training_data.csv...")
try:
    df = pd.read_csv('final_combined_training_data.csv')
    print("Data loaded successfully.")
    print(f"Initial DataFrame shape: {df.shape}")
    print("Initial DataFrame columns:", df.columns.tolist())
except FileNotFoundError:
    print("Error: 'final_combined_training_data.csv' not found. Please ensure the file is in the correct directory.")
    exit() # Exit the script if the file isn't found

# --- 2. Data Preprocessing ---

# Clean the 'burnout' column (target variable)
print("\nCleaning 'burnout' column...")
df['burnout'] = df['burnout'].replace({
    'Not at all': 0, 'Mild tiredness': 1, 'Drained': 2, 'Full-on burnout': 3
})
# Convert to numeric, coercing any remaining non-numeric values to NaN
df['burnout'] = pd.to_numeric(df['burnout'], errors='coerce')
print(f"Burnout column unique values after initial cleaning: {df['burnout'].unique()}")

# --- Robustly convert specific columns to numeric, coercing errors to NaN ---
# These are columns that should be numerical, but might contain strings or mixed types
columns_to_force_numeric = [
    'age', 'mood', 'anxiety', 'energy', 'sleep',
    'refreshed_after_sleep', 'steps_taken', 'water_intake',
    'caffeine_intake', 'work_hours'
]

print("\nConverting specified columns to numeric type...")
for col in columns_to_force_numeric:
    if col in df.columns:
        initial_dtype = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if initial_dtype != df[col].dtype:
            print(f"Converted '{col}' from {initial_dtype} to {df[col].dtype}.")
    else:
        print(f"Warning: Numeric column '{col}' not found in DataFrame. Skipping conversion.")

# --- Handle 'unknown' in user profile categories (if they exist) ---
user_profile_columns = [
    'occupation', 'alcohol_consumption', 'prior_burnout_anxiety',
    'living_situation', 'supportive_environment',
    'liking_study_work_environment', 'contraceptive_pill_use',
    'drug_use_smoking_habits'
]
# Filter for actual columns present in df
user_profile_columns_present = [col for col in user_profile_columns if col in df.columns]

if user_profile_columns_present:
    print("\nHandling 'Unknown' for user profile columns...")
    for col in user_profile_columns_present:
        if df[col].dtype == 'object': # Only fill NaNs if it's an object type column
            df[col] = df[col].fillna('Unknown')
    print(f"Processed user profile columns: {user_profile_columns_present}")
else:
    print("\nNo user profile columns found in DataFrame to process.")


# --- Explicitly convert binary 'Yes'/'No' columns to 0/1 integers ---
print("\nConverting binary 'Yes'/'No' columns to 0/1 integers...")
binary_map = {'Yes': 1, 'No': 0, 'Unknown': 0, 1.0:1, 0.0:0} # Added 1.0/0.0 for robustness
# List of columns that are expected to be binary 'Yes'/'No' strings or similar
binary_cols_to_convert = [
    'on_period_today', 'has_sad_pop', 'has_low_energy',
    'has_food_cravings', 'has_cramps', 'has_headache'
]
# Filter for actual columns present in df
binary_cols_present = [col for col in binary_cols_to_convert if col in df.columns]

if binary_cols_present:
    for col in binary_cols_present:
        # Convert to string first to ensure .map works consistently, then map
        df[col] = df[col].astype(str).map(binary_map).fillna(0).astype(int)
        print(f"Converted '{col}' to 0/1 integers.")
else:
    print("\nNo binary 'Yes'/'No' columns found in DataFrame to convert.")


# --- Identify all categorical columns for One-Hot Encoding ---
# These are columns that remain as 'object' type after initial numeric conversions and binary mappings
# Exclude columns that will be dropped later (like 'date', 'user_id', etc.)
# Exclude the target column 'burnout'
columns_to_exclude_from_ohe = [
    'burnout', 'date', 'user_id', 'journal_entry', 'major_event_log'
]
categorical_cols_for_ohe_present = [
    col for col in df.columns
    if df[col].dtype == 'object' and col not in columns_to_exclude_from_ohe
]

# Ensure known categorical columns are explicitly in the list if they are present and still objects
# This covers 'music_genre', 'music_time', 'outfit_type', 'music_volume', 'cycle_phase', 'symptoms_experienced'
# and also any user profile columns that might still be objects.
explicit_categorical_cols = [
    'outfit_type', 'music_volume', 'cycle_phase', 'music_genre', 'music_time', 'symptoms_experienced'
]
explicit_categorical_cols.extend(user_profile_columns_present) # Add present user profile cols

# Combine and remove duplicates for the final list
categorical_cols_for_ohe_final = list(set(categorical_cols_for_ohe_present + explicit_categorical_cols))
categorical_cols_for_ohe_final = [col for col in categorical_cols_for_ohe_final if col in df.columns and df[col].dtype == 'object']


if categorical_cols_for_ohe_final:
    print(f"\nApplying One-Hot Encoding for the following categorical features: {categorical_cols_for_ohe_final}")
    # pd.get_dummies will drop original columns if 'columns' argument is used
    df_processed_for_ml = pd.get_dummies(df, columns=categorical_cols_for_ohe_final, drop_first=False)
    print(f"DataFrame shape after One-Hot Encoding: {df_processed_for_ml.shape}")
else:
    df_processed_for_ml = df.copy() # If no categories, just copy
    print("\nNo remaining categorical columns found for One-Hot Encoding.")


# --- 3. Define Target Variable and Features ---
TARGET_COLUMN = 'burnout'

# Drop rows where the target variable is NaN (after initial cleaning and conversion)
initial_rows = len(df_processed_for_ml)
df_processed_for_ml.dropna(subset=[TARGET_COLUMN], inplace=True)
rows_after_nan_drop = len(df_processed_for_ml)
if initial_rows - rows_after_nan_drop > 0:
    print(f"\nDropped {initial_rows - rows_after_nan_drop} rows due to NaN in '{TARGET_COLUMN}' column after conversion.")
    print(f"Remaining rows: {rows_after_nan_drop}")

# Assign target variable y
y = df_processed_for_ml[TARGET_COLUMN].astype(int)

# Define X based on df_processed_for_ml.
# Exclude 'date', 'user_id', 'journal_entry', 'major_event_log', and the target column itself.
columns_to_drop_from_X = [TARGET_COLUMN, 'date', 'user_id', 'journal_entry', 'major_event_log']
X = df_processed_for_ml.drop(columns=columns_to_drop_from_X, errors='ignore')

# Align X with y by index
X = X.loc[y.index]

print(f"\nFeatures (X) shape after final selection: {X.shape}")
print(f"Target (y) shape after final selection: {y.shape}")
print("\nNull counts in X before ColumnTransformer:")
print(X.isnull().sum()[X.isnull().sum() > 0]) # Print only columns with nulls
print("\nObject (string) columns in X before ColumnTransformer (should be empty):")
print(X.select_dtypes(include='object').columns.tolist())


# --- 4. Column Transformer Setup for numerical features ---
# The numerical features are those that remain float64 after all previous conversions and are not dummy variables.
# We will use numerical_features_for_ct to explicitly select numerical columns.
# This list now contains all original numericals and the 0/1 binary columns.
numerical_features_for_ct = [
    'age', 'mood', 'anxiety', 'energy', 'sleep',
    'refreshed_after_sleep', 'steps_taken', 'water_intake',
    'caffeine_intake', 'work_hours'
]
numerical_features_for_ct.extend(binary_cols_to_convert) # Add the binary columns that are now 0/1

# Filter for actual columns present in X and are numerical
features_to_scale_actual = [
    col for col in numerical_features_for_ct
    if col in X.columns and pd.api.types.is_numeric_dtype(X[col])
]

if not features_to_scale_actual:
    print("Warning: No numerical features found for scaling. Check 'numerical_features_for_ct' list and DataFrame columns.")
else:
    print(f"\nNumerical features to be scaled and imputed: {features_to_scale_actual}")

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For ColumnTransformer, we define transformers for specific numerical features
# and let 'remainder='passthrough'' handle all other columns (which should now be one-hot encoded and numerical).
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, features_to_scale_actual)
    ],
    remainder='passthrough' # This will pass through all other columns that are not in features_to_scale_actual
)

# --- 5. Apply Preprocessing ---
print("\nApplying final preprocessing pipeline (imputation, scaling) on X...")
X_processed = preprocessor.fit_transform(X)

# Get feature names after preprocessing
# This handles the 'num__' and 'remainder__' prefixes from ColumnTransformer
all_feature_names = preprocessor.get_feature_names_out()

# Convert back to DataFrame
# Check if it's a sparse matrix (e.g., if one-hot encoding created many zeros)
if hasattr(X_processed, 'toarray'):
    X_processed_df = pd.DataFrame(X_processed.toarray(), columns=all_feature_names, index=X.index)
else:
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)

print(f"\nProcessed features (X_processed_df) shape: {X_processed_df.shape}")
print("Processed DataFrame Info:")
X_processed_df.info()
print("\nFirst 5 rows of processed features:")
print(X_processed_df.head())


# --- 6. Split Data into Training and Test Sets ---
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 7. Model Selection and Training (RandomForestClassifier) ---
print("\nTraining a RandomForestClassifier model...")
# Increased n_estimators for potentially better performance, added class_weight to handle potential imbalance
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Model training complete.")

# --- 8. Model Evaluation ---
print("\nEvaluating the model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- 9. Save the Trained Model and Preprocessor ---
MODEL_SAVE_PATH = 'burnout_prediction_random_forest_model.pkl'
PREPROCESSOR_SAVE_PATH = 'data_preprocessor_for_random_forest.pkl'

joblib.dump(model, MODEL_SAVE_PATH)
joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)

print(f"\nModel saved to {MODEL_SAVE_PATH}")
print(f"Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")

print("\nML pipeline execution complete with RandomForestClassifier. Check the accuracy above.")