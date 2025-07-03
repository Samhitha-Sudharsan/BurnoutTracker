import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Import Hugging Face libraries
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import joblib

# Import feature engineering functions from your processor
from burnout_data_processor import create_domain_specific_features, create_rolling_features

# --- GLOBAL MODEL INITIALIZATION (for NLP features during training) ---
# These models are initialized ONLY ONCE when train_burnout_model.py is executed.
print("Initializing global sentiment analysis pipeline...")
global_sentiment_pipeline = pipeline("sentiment-analysis")
print("Global sentiment analysis pipeline loaded.")

print("Initializing global SentenceTransformer model...")
global_sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Global SentenceTransformer model loaded.")
# --- END GLOBAL MODEL INITIALIZATION ---


def create_sentiment_features(df):
    """
    Creates sentiment scores and sentence embeddings from text fields.
    Uses globally defined sentiment models.
    """
    global global_sentiment_pipeline, global_sentence_model

    # Combine text fields into a single 'combined_text' column BEFORE NLP processing
    if 'journal_entry' in df.columns and 'major_event_log' in df.columns and 'symptoms_experienced' in df.columns:
        df['combined_text'] = df['journal_entry'].fillna('') + " " + \
                              df['major_event_log'].fillna('') + " " + \
                              df['symptoms_experienced'].fillna('')
        df['combined_text'] = df['combined_text'].str.strip().replace(r'\s+', ' ', regex=True) # Clean up multiple spaces
    elif 'combined_text' not in df.columns: # Fallback if individual text columns aren't present
        df['combined_text'] = '' # Ensure combined_text column exists even if empty

    if 'combined_text' in df.columns:
        # Fill NaN with empty string to avoid errors in NLP models
        df['combined_text_clean'] = df['combined_text'].astype(str).replace('nan', '')

        # Apply sentiment analysis
        sentiment_results = [
            global_sentiment_pipeline(x)[0] if x else {'score': np.nan, 'label': 'neutral'}
            for x in df['combined_text_clean']
        ]
        df['sentiment_score'] = [s['score'] for s in sentiment_results]
        df['sentiment_label'] = [s['label'] for s in sentiment_results]

        # Apply sentence embeddings
        non_empty_text_indices = df[df['combined_text_clean'].str.len() > 0].index
        if not non_empty_text_indices.empty:
            valid_texts = df.loc[non_empty_text_indices, 'combined_text_clean'].tolist()
            embeddings = global_sentence_model.encode(valid_texts)
            # Create a temporary Series for embeddings to align indices
            embedding_series = pd.Series(list(embeddings), index=non_empty_text_indices)
            # Assign to a new column, ensuring it's object dtype
            df['text_embedding'] = pd.Series(dtype=object) # Initialize with object dtype
            df.loc[non_empty_text_indices, 'text_embedding'] = embedding_series
        else:
            df['text_embedding'] = [np.nan] * len(df) # Fill with NaNs if no valid text

        df.drop(columns=['combined_text_clean'], inplace=True)
    return df
# --- END FEATURE ENGINEERING FUNCTIONS ---


def train_model(TARGET_USER_ID):
    """
    Loads user data, applies feature engineering and preprocessing,
    and trains a RandomForestClassifier using GridSearchCV.
    """
    # Assuming 'final_combined_training_data.csv' is your comprehensive raw data source
    df = pd.read_csv('final_combined_training_data.csv')

    # Filter data for the specific user
    df_user = df[df['user_id'] == TARGET_USER_ID].copy()

    if df_user.empty:
        print(f"No data found for user ID: {TARGET_USER_ID}")
        return None, None

    # Sort by date for rolling features
    df_user['date'] = pd.to_datetime(df_user['date'], errors='coerce')
    df_user = df_user.sort_values(by='date').reset_index(drop=True)

    # Convert intake columns to numerical using the processor function
    # NOTE: The create_domain_specific_features function now handles this correctly by calling convert_intake_to_numeric_value internally.
    # So you don't need these explicit calls here anymore if create_domain_specific_features is updated.
    # For now, keeping as is, but consider consolidating.

    # Apply feature engineering functions in sequence
    # order matters: domain_specific first to convert categorical columns needed by rolling
    df_user = create_domain_specific_features(df_user)
    df_user = create_rolling_features(df_user)
    df_user = create_sentiment_features(df_user) # This also creates 'combined_text' if needed

    # --- Start of Critical Target Variable Handling ---
    if 'burnout' in df_user.columns:
        df_user['burnout'] = pd.to_numeric(df_user['burnout'], errors='coerce')
        initial_rows = len(df_user)
        df_user.dropna(subset=['burnout'], inplace=True)
        if len(df_user) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(df_user)} rows due to NaN in 'burnout' for user {TARGET_USER_ID}.")
        df_user['burnout'] = df_user['burnout'].astype(int)
    else:
        print(f"Error: 'burnout' column not found in data for user {TARGET_USER_ID}.")
        return None, None
    # --- End of Critical Target Variable Handling ---


    # --- Enhanced Pre-Imputation (before pipeline to handle all-NaN columns) ---
    # These lists should be comprehensive for all expected input features for the CT
    # This assumes 'sleep', 'mood', 'anxiety', 'energy', 'age', 'work_hours', 'steps_taken', 'music_time' are direct numerical inputs
    numerical_cols_to_check = [
        'sleep', 'mood', 'anxiety', 'energy', 'age', 'work_hours', 'steps_taken', 'music_time',
        'water_intake', 'caffeine_intake',
        'contraceptive_pill_use_encoded', 'prior_history_burnout_encoded',
        'prior_history_anxiety_encoded', 'drug_use_encoded', 'smoking_habits_encoded',
        'alcohol_consumption_encoded', 'supportive_environment_encoded',
        'liking_study_work_environment_encoded',
        'insufficient_sleep', 'excessive_sleep', 'is_sleep_deprived',
        'low_mood', 'high_anxiety', 'low_energy',
        'heavy_alcohol_consumption', 'drug_use_smoking_habits', 'drug_use_or_smoking_present',
        'low_mood_luteal_interaction', 'low_energy_luteal_interaction',
        'prior_issues_interaction',
        'mood_3day_avg', 'mood_3day_std', 'mood_7day_avg', 'mood_7day_std',
        'sleep_3day_avg', 'sleep_3day_std', 'sleep_7day_avg', 'sleep_7day_std',
        'anxiety_3day_avg', 'anxiety_3day_std', 'anxiety_7day_avg', 'anxiety_7day_std',
        'energy_3day_avg', 'energy_3day_std', 'energy_7day_avg', 'energy_7day_std',
        'sentiment_score' # Numeric sentiment score
    ]

    categorical_cols_to_check = [
        'refreshed_after_sleep', 'outfit_type', 'music_genre', 'music_volume',
        'on_period_today', 'cycle_phase', 'occupation', 'living_situation',
        'sentiment_label' # Categorical sentiment label
    ]

    for col in numerical_cols_to_check:
        if col in df_user.columns:
            if df_user[col].isnull().all():
                df_user[col] = df_user[col].fillna(0.0) # Fill with 0 for numerical all-NaNs

    for col in categorical_cols_to_check:
        if col in df_user.columns:
            if df_user[col].isnull().all():
                df_user[col] = df_user[col].fillna('__MISSING__') # Fill with a placeholder string for categorical all-NaNs
    # --- End of Enhanced Pre-Imputation ---

    # Define features (X) and target (y)
    # Ensure 'combined_text' is dropped from X if you are expanding 'text_embedding'
    # 'journal_entry', 'major_event_log', 'symptoms_experienced' should also be dropped
    cols_to_drop_from_X = ['user_id', 'date', 'burnout',
                           'journal_entry', 'major_event_log', 'symptoms_experienced', # Raw text fields
                           'combined_text' # The combined text field itself, after embeddings
                          ]
    X = df_user.drop(columns=[col for col in cols_to_drop_from_X if col in df_user.columns], errors='ignore')
    y = df_user['burnout']

    # --- START: Handling 'text_embedding' expansion ---
    # NOTE: Ensure global_sentence_model is loaded if you run this outside __main__ or as a script
    EMBEDDING_DIMENSION = global_sentence_model.get_sentence_embedding_dimension() if global_sentence_model else 384 # Default to 384 if not loaded

    if 'text_embedding' in X.columns:
        valid_embeddings = X['text_embedding'].dropna()
        if not valid_embeddings.empty:
            embeddings_df = pd.DataFrame(valid_embeddings.tolist(), index=valid_embeddings.index)
            # Ensure column names are unique and reflect the embedding dimension
            embeddings_df.columns = [f'text_embedding_{i}' for i in range(embeddings_df.shape[1])]
            X = pd.concat([X.drop(columns=['text_embedding']), embeddings_df], axis=1)
        else:
            # If 'text_embedding' exists but is all NaN, drop it and add zero vectors
            X = X.drop(columns=['text_embedding'])
            zero_embeddings_df = pd.DataFrame(0.0, index=X.index, columns=[f'text_embedding_{i}' for i in range(EMBEDDING_DIMENSION)])
            X = pd.concat([X, zero_embeddings_df], axis=1)
    else:
        # If 'text_embedding' was never created, add zero vectors
        zero_embeddings_df = pd.DataFrame(0.0, index=X.index, columns=[f'text_embedding_{i}' for i in range(EMBEDDING_DIMENSION)])
        X = pd.concat([X, zero_embeddings_df], axis=1)
    # --- END: Handling 'text_embedding' expansion ---


    # Identify numerical and categorical features for ColumnTransformer after embedding expansion
    # This is important: re-select columns after embedding expansion
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any columns not explicitly transformed (e.g., if there are new ones)
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X, y)

    print(f"\n--- Results for User: {TARGET_USER_ID} ---")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    return best_model, X.columns.tolist() # Return as list for easier saving/consistency


# --- Main execution block for train_burnout_model.py ---
if __name__ == "__main__":
    TARGET_USER_ID = 'real_user_000' # Example user ID for training data
    print(f"\nStarting model training for user: {TARGET_USER_ID}")

    trained_model, feature_names_at_fit = train_model(TARGET_USER_ID)

    if trained_model:
        print(f"\nModel for user {TARGET_USER_ID} trained successfully.")
        
        # --- IMPORTANT: Standardized Model Saving Path ---
        MODEL_PATH_FOR_SAVE = 'burnout_prediction_random_forest_model_tuned_with_bert_nlp_and_domain_features_v2.pkl'
        joblib.dump(trained_model, MODEL_PATH_FOR_SAVE)
        print(f"Model saved to: {MODEL_PATH_FOR_SAVE}")

        # It's good practice to save the feature names the model was trained on
        # You might not use this in predict.py directly, but it's invaluable for debugging
        # or verifying inputs if the pipeline changes.
        FEATURE_NAMES_PATH_FOR_SAVE = 'feature_names_for_burnout_model_v2.pkl'
        joblib.dump(feature_names_at_fit, FEATURE_NAMES_PATH_FOR_SAVE)
        print(f"Feature names saved to: {FEATURE_NAMES_PATH_FOR_SAVE}")

    else:
        print(f"Failed to train model for user {TARGET_USER_ID}.")