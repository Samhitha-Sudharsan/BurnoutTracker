import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, pipeline

# Define BERT model name globally for consistency
BERT_MODEL_NAME = 'bert-base-uncased'

# Load the pre-trained BERT tokenizer globally
try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
except Exception as e:
    print(f"Error loading BERT tokenizer: {e}")
    tokenizer = None

# Load the BERT encoder model globally
try:
    bert_model = TFAutoModel.from_pretrained(BERT_MODEL_NAME)
except Exception as e:
    print(f"Error loading TFAutoModel (BERT): {e}")
    print("Please ensure you have tensorflow installed and compatible with transformers.")
    bert_model = None

# Initialize sentiment analysis pipeline globally for efficiency
sentiment_pipeline = None
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
except Exception as e:
    print(f"Error loading sentiment analysis pipeline: {e}")
    print("Sentiment analysis will not be available. Please check your internet connection or Hugging Face setup.")

def get_bert_embeddings(text_series, max_seq_length=128):
    if tokenizer is None or bert_model is None:
        print("BERT tokenizer or model not loaded, returning empty embeddings.")
        return np.zeros((len(text_series), 768)) # BERT-base output dimension

    text_series = text_series.fillna('').astype(str)

    inputs = tokenizer(
        text_series.tolist(),
        max_length=max_seq_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    outputs = bert_model(**inputs)
    pooled_output = outputs.last_hidden_state[:, 0, :] # Use CLS token embedding

    return pooled_output.numpy()

def convert_intake_to_numeric_value(intake_str, intake_type):
    """
    Converts categorical intake strings to numerical values.
    """
    if pd.isna(intake_str):
        return np.nan

    if intake_type == 'water_intake':
        if intake_str == "0-0.5L": return 0.25
        elif intake_str == "0.5-1.5L": return 1.0
        elif intake_str == "1.5-2.5L": return 2.0
        elif intake_str == "2.5-3.5L": return 3.0
        elif intake_str == "3.5L+": return 4.0
    elif intake_type == 'caffeine_intake':
        if intake_str == "None": return 0
        elif intake_str == "1 cup": return 1
        elif intake_str == "2 cups": return 2
        elif intake_str == "3 cups": return 3
        elif intake_str == "4+ cups": return 4
        # Add any other specific caffeine categories you might have
    elif intake_type == 'alcohol_consumption':
        if intake_str == "None": return 0
        elif intake_str == "1-2 drinks/day (occasional/moderate)": return 1.5
        elif intake_str == "More than 2 drinks/day (regular/heavy)": return 3.0
    # Add other intake types if needed
    return np.nan # Return NaN for unhandled or invalid strings

def process_data_for_prediction(current_day_raw_log, latest_user_profile, historical_daily_logs, expected_columns=None):
    if current_day_raw_log.empty or latest_user_profile.empty:
        return pd.DataFrame(), "Error: Daily log or user profile is empty."

    # Ensure date columns are datetime objects
    current_day_raw_log['date'] = pd.to_datetime(current_day_raw_log['date'], errors='coerce')
    historical_daily_logs['date'] = pd.to_datetime(historical_daily_logs['date'], errors='coerce')

    user_id = current_day_raw_log['user_id'].iloc[0]
    user_historical_logs = historical_daily_logs[
        historical_daily_logs['user_id'] == user_id
    ].sort_values(by='date')

    df_combined = current_day_raw_log.copy()

    # --- Feature Engineering ---
    df_combined['day_of_week'] = df_combined['date'].dt.dayofweek
    df_combined['month'] = df_combined['date'].dt.month

    df_combined['day_of_week_sin'] = np.sin(2 * np.pi * df_combined['day_of_week'] / 6)
    df_combined['day_of_week_cos'] = np.cos(2 * np.pi * df_combined['day_of_week'] / 6)
    df_combined['month_sin'] = np.sin(2 * np.pi * df_combined['month'] / 11)
    df_combined['month_cos'] = np.cos(2 * np.pi * df_combined['month'] / 11)

    # Combine historical and current for rolling features
    all_logs_for_user = pd.concat([user_historical_logs, current_day_raw_log]).drop_duplicates(subset=['date', 'user_id']).sort_values(by='date')

    for col in ['sleep', 'mood', 'anxiety', 'energy', 'work_hours', 'steps_taken', 'music_time']:
        if col in all_logs_for_user.columns:
            all_logs_for_user[col] = pd.to_numeric(all_logs_for_user[col], errors='coerce')
            # Use .iloc[-1] to get the value for the *current* log after rolling mean calculation
            df_combined[f'{col}_7day_avg'] = all_logs_for_user[col].rolling(window=7, min_periods=1).mean().iloc[-1]
        else:
            df_combined[f'{col}_7day_avg'] = df_combined[col].iloc[0] if col in df_combined.columns else 0 # Default to 0 if column is completely missing

    if len(all_logs_for_user) > 1:
        prev_day_log = all_logs_for_user.iloc[-2]
        for col in ['sleep', 'mood', 'anxiety', 'energy', 'work_hours']:
            if col in all_logs_for_user.columns:
                current_val = pd.to_numeric(df_combined[col].iloc[0], errors='coerce')
                prev_val = pd.to_numeric(prev_day_log[col], errors='coerce')
                df_combined[f'{col}_change'] = current_val - prev_val if not pd.isna(current_val) and not pd.isna(prev_val) else 0
            else:
                df_combined[f'{col}_change'] = 0
    else:
        for col in ['sleep', 'mood', 'anxiety', 'energy', 'work_hours']:
            df_combined[f'{col}_change'] = 0

    # --- BERT Embeddings for Journal Entry and Major Event Log ---
    journal_entries = df_combined['journal_entry'].fillna('').astype(str)
    major_events = df_combined['major_event_log'].fillna('').astype(str)

    journal_embedding = get_bert_embeddings(journal_entries)
    major_event_embedding = get_bert_embeddings(major_events)

    for i in range(journal_embedding.shape[1]):
        df_combined[f'journal_embedding_{i}'] = journal_embedding[:, i]
    for i in range(major_event_embedding.shape[1]):
        df_combined[f'major_event_embedding_{i}'] = major_event_embedding[:, i]

    # --- Sentiment Analysis ---
    combined_text = (journal_entries + " " + major_events).replace(r'^\s*$', np.nan, regex=True).fillna('')

    if sentiment_pipeline and not combined_text.empty and combined_text.iloc[0] != '':
        sentiment_results = sentiment_pipeline(combined_text.tolist())
        df_combined['sentiment_label'] = sentiment_results[0]['label']
        df_combined['sentiment_score'] = sentiment_results[0]['score']
    else:
        df_combined['sentiment_label'] = 'Neutral' # Default if pipeline fails or no text
        df_combined['sentiment_score'] = 0.5
        if not sentiment_pipeline:
            print("Warning: Sentiment analysis pipeline not loaded. Using default sentiment.")

    sentiment_dummies = pd.get_dummies(df_combined['sentiment_label'], prefix='sentiment')
    df_combined = pd.concat([df_combined, sentiment_dummies], axis=1)
    for label in ['sentiment_Positive', 'sentiment_Negative', 'sentiment_Neutral']:
        if label not in df_combined.columns:
            df_combined[label] = 0

    # --- Merge with User Profile Data ---
    df_combined['user_id'] = df_combined['user_id'].astype(str)
    latest_user_profile['user_id'] = latest_user_profile['user_id'].astype(str)

    profile_df_for_merge = latest_user_profile.reset_index(drop=True)
    df_combined = pd.merge(df_combined, profile_df_for_merge, on='user_id', how='left')

    # --- One-Hot Encoding for Categorical Features ---
    all_possible_categories = {
        'refreshed_after_sleep': ['Yes', 'No'],
        'music_genre': ['Pop', 'Rock', 'Classical', 'Jazz', 'Electronic', 'Other', 'None', 'Not Tracked'],
        'music_volume': ['Low', 'Medium', 'High', 'Not Applicable', 'Not Tracked'],
        'on_period_today': ['Yes', 'No'], # Ensure consistent with app's "Yes"/"No" storage
        'cycle_phase': ['Menstrual', 'Follicular', 'Ovulatory', 'Luteal', 'Unknown', 'Period'],
        'water_intake': ['0-0.5L', '0.5-1.5L', '1.5-2.5L', '2.5-3.5L', '3.5L+'],
        'caffeine_intake': ['None', '1 cup', '2 cups', '3 cups', '4+ cups'], # Removed extra categories not in app form
        'alcohol_consumption': ['None', '1-2 drinks/day (occasional/moderate)', 'More than 2 drinks/day (regular/heavy)'],
        'drug_use': ['No', 'Yes', 'Prefer not to say'],
        'smoking_habits': ['No', 'Yes', 'Prefer not to say'],
        'occupation': ['Student', 'Professional', 'Homemaker', 'Unemployed', 'Other'], # Consider adding more if app allows free text
        'living_situation': ['Alone', 'With Partner', 'With Roommates', 'With Family'], # Consider adding more if app allows free text
        'contraceptive_pill_use': ['No', 'Yes', 'Prefer not to say'],
        'prior_history_burnout': ['No', 'Yes', 'Prefer not to say'],
        'prior_history_anxiety': ['No', 'Yes', 'Prefer not to say'],
        'supportive_environment': ['Yes', 'No'],
        'liking_study_work_environment': ['Yes', 'No'],
        'outfit_type': ['Casual', 'Formal', 'Smart Casual', 'Loungewear', 'Sporty']
    }

    for col, categories in all_possible_categories.items():
        if col in df_combined.columns:
            # Handle potential NaN values and convert to string before one-hot encoding
            df_combined[col] = df_combined[col].astype(str)
            df_combined[col] = pd.Categorical(df_combined[col], categories=categories)
            dummies = pd.get_dummies(df_combined[col], prefix=col)
            df_combined = pd.concat([df_combined, dummies], axis=1)
            df_combined.drop(columns=[col], inplace=True)
        else:
            # If the column is completely missing from the dataframe, ensure all its dummy columns are 0
            for cat in categories:
                df_combined[f'{col}_{cat}'] = 0

    df_processed = df_combined.drop(columns=[
        'user_id', 'date', 'journal_entry', 'major_event_log', 'symptoms_experienced', 'sentiment_label'
    ], errors='ignore')

    # Ensure 'age' is numeric
    if 'age' in df_processed.columns:
        df_processed['age'] = pd.to_numeric(df_processed['age'], errors='coerce').fillna(df_processed['age'].mean() if not df_processed['age'].empty else 0)
    else:
        df_processed['age'] = 0

    # Reindex to match expected columns from the model if provided
    if expected_columns is not None:
        # Add missing columns with 0
        for col in expected_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        # Drop extra columns and reorder
        df_processed = df_processed[expected_columns]

    return df_processed, None