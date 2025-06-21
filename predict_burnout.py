import joblib
import pandas as pd
import json
import numpy as np # Make sure numpy is imported for median() if you use it

# Load the trained model and preprocessor
model = joblib.load('burnout_prediction_model.pkl')
preprocessor = joblib.load('data_preprocessor.pkl')

# Load the comprehensive lists of genres and symptoms from training data
with open('all_music_genres.json', 'r') as f:
    ALL_TRAINING_MUSIC_GENRES = json.load(f)

with open('all_symptoms.json', 'r') as f:
    ALL_TRAINING_SYMPTOMS = json.load(f)

def predict_new_data(new_data_df):
    """
    Predicts burnout level for new, raw DataFrame of data.
    new_data_df should have the same columns as your original training data,
    excluding 'burnout', 'date', 'user_id', 'journal_entry'.
    """
    # Create a copy to avoid SettingWithCopyWarning
    new_data_df = new_data_df.copy()

    # 1. Apply the same manual binarization for multi-select features
    # This must be consistent with how training data was prepared

    # Handle music_genre
    if 'music_genre' in new_data_df.columns:
        new_data_df['music_genre'] = new_data_df['music_genre'].fillna('')
        # Create binarized columns for ALL genres seen during training
        for genre in ALL_TRAINING_MUSIC_GENRES:
            # Use .apply and assign back for Series (avoids FutureWarning)
            new_data_df[f'music_genre_{genre.replace(" ", "_").replace("-", "_")}'] = \
                new_data_df['music_genre'].apply(lambda x: 1 if genre in x else 0)
        new_data_df = new_data_df.drop(columns=['music_genre'])

    # Handle symptoms_experienced
    if 'symptoms_experienced' in new_data_df.columns:
        new_data_df['symptoms_experienced'] = new_data_df['symptoms_experienced'].fillna('')
        # Create binarized columns for ALL symptoms seen during training
        for symptom in ALL_TRAINING_SYMPTOMS:
            # Use .apply and assign back for Series (avoids FutureWarning)
            new_data_df[f'symptoms_{symptom.replace(" ", "_").replace("-", "_")}'] = \
                new_data_df['symptoms_experienced'].apply(lambda x: 1 if symptom in x else 0)
        new_data_df = new_data_df.drop(columns=['symptoms_experienced'])

    # 2. Apply the same ordinal mapping for X features
    ordinal_features_for_X_mapping = {
        'refreshed_after_sleep': ['No', 'Kind of', 'Yes'],
        'steps_taken': ['<2000', '2000–5000', '5000–8000', '8000+'],
        'water_intake': ['<1L', '1–1.5L', '1.5–2.5L', '2.5L+'],
        'caffeine_intake': ['None', '1 cup', '2 cups', '3+ cups'],
        'work_hours': ['<2 hrs', '2–5 hrs', '5–8 hrs', '8+ hrs']
    }
    for feature, order in ordinal_features_for_X_mapping.items():
        if feature in new_data_df.columns:
            mapping_dict = {label: i for i, label in enumerate(order)}
            # Assign result back to avoid FutureWarning
            new_data_df[feature] = new_data_df[feature].map(mapping_dict)
            # Handle potential NaNs from mapping for new data by imputing or dropping
            # For prediction, imputation is usually preferred so you don't drop rows
            # Assign result back to avoid FutureWarning
            if new_data_df[feature].isnull().any(): # Only impute if there are NaNs
                 new_data_df[feature] = new_data_df[feature].fillna(new_data_df[feature].median()) # or use a specific value like 0

    # 3. Ensure 'major_event_log' is dropped if it exists in new data
    if 'major_event_log' in new_data_df.columns:
        new_data_df = new_data_df.drop(columns=['major_event_log'])

    # 4. Use the preprocessor to transform the new data
    # IMPORTANT: Use .transform(), NOT .fit_transform()
    X_new_processed = preprocessor.transform(new_data_df)

    # Convert to DataFrame with correct column names (optional but good for inspection)
    feature_names_out = preprocessor.get_feature_names_out()
    if hasattr(X_new_processed, 'toarray'):
        X_new_processed_df = pd.DataFrame(X_new_processed.toarray(), columns=feature_names_out)
    else:
        X_new_processed_df = pd.DataFrame(X_new_processed, columns=feature_names_out)


    # 5. Make predictions
    predictions = model.predict(X_new_processed_df)

    # Optionally, map predictions back to original labels if needed
    inverse_burnout_mapping = {
        0: "Not at all",
        1: "Mild tiredness",
        2: "Drained",
        3: "Full-on burnout"
    }
    predicted_labels = [inverse_burnout_mapping[p] for p in predictions]

    return predicted_labels

# Example usage with dummy data
# This dummy data MUST mirror the structure of your original training data's features
# (before ColumnTransformer but AFTER your initial manual binarization and ordinal mapping for X).
# It's crucial that any new data has the same columns and types as what the preprocessor saw during training.
dummy_new_data = pd.DataFrame({
    'age': [25, 30],
    'mood': [3, 2],
    'anxiety': [1, 4],
    'energy': [4, 1],
    'sleep': [7, 5],
    'refreshed_after_sleep': ['Yes', 'No'],
    'steps_taken': ['5000–8000', '<2000'],
    'water_intake': ['1.5–2.5L', '<1L'],
    'caffeine_intake': ['1 cup', '3+ cups'],
    'work_hours': ['5–8 hrs', '8+ hrs'],
    'outfit_type': ['Casual (jeans, tee)', 'Athletic/Activewear'],
    'music_volume': ['Medium', 'Loud'],
    'music_time': ['Morning', 'Night'],
    'on_period_today': ['No', 'Yes'],
    'cycle_phase': ['Follicular', 'Luteal'],
    'music_genre': ['Pop, Rock', 'EDM'], # Original format
    'symptoms_experienced': ['Low energy', 'Cramps, Food craving'], # Original format
    # 'major_event_log': ['Some event', 'Another event'] # Should be dropped if present
})

predicted_burnout = predict_new_data(dummy_new_data)
print(f"\nPredicted burnout levels for new data: {predicted_burnout}")