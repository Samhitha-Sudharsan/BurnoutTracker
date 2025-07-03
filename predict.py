import pandas as pd
import pickle
import os
import numpy as np
# No longer need tensorflow, AutoTokenizer, TFAutoModel, pipeline directly here
# because they are now handled within burnout_data_processor.py

# Import the data processing function and relevant components from burnout_data_processor
from burnout_data_processor import process_data_for_prediction, convert_intake_to_numeric_value # Keep convert_intake for advice logic if needed
# Note: get_bert_embeddings, sentiment_pipeline are handled internally by process_data_for_prediction now.

# Define paths (keep these for the model itself)
MODEL_PATH = 'burnout_prediction_random_forest_model_tuned_with_bert_nlp_and_domain_features_v2.pkl'

# Load the pre-trained Random Forest model
full_model_pipeline = None
expected_model_features = None # To store feature names the model expects
try:
    with open(MODEL_PATH, 'rb') as f:
        full_model_pipeline = pickle.load(f)
        # Attempt to extract feature names the model expects
        if full_model_pipeline is not None:
            if hasattr(full_model_pipeline, 'feature_names_in_'):
                expected_model_features = full_model_pipeline.feature_names_in_
            elif hasattr(full_model_pipeline, 'named_steps') and 'randomforestclassifier' in full_model_pipeline.named_steps and hasattr(full_model_pipeline.named_steps['randomforestclassifier'], 'feature_names_in_'):
                expected_model_features = full_model_pipeline.named_steps['randomforestclassifier'].feature_names_in_
            elif hasattr(full_model_pipeline, 'steps'): # For older sklearn Pipelines
                # This is more complex, might need to inspect the last step
                last_step_model = full_model_pipeline.steps[-1][1]
                if hasattr(last_step_model, 'feature_names_in_'):
                    expected_model_features = last_step_model.feature_names_in_
    if expected_model_features is None:
        print("Warning: Could not extract expected feature names from the loaded model. This might lead to prediction errors if feature order/presence is inconsistent.")

except FileNotFoundError:
    print(f"Warning: Model file not found at {MODEL_PATH}. Prediction will not be available.")
except Exception as e:
    print(f"Error loading the pickled model: {e}")


def predict_burnout(current_day_raw_log, latest_user_profile, historical_daily_logs):
    if full_model_pipeline is None:
        return {"predicted_label": "Error: Model not loaded.", "predicted_level": -1, "probabilities": {}}

    # Call the processing function from burnout_data_processor
    processed_features, error = process_data_for_prediction(
        current_day_raw_log=current_day_raw_log,
        latest_user_profile=latest_user_profile,
        historical_daily_logs=historical_daily_logs,
        expected_columns=expected_model_features # Pass the expected features from the loaded model
    )

    if error:
        return {"predicted_label": error, "predicted_level": -1, "probabilities": {}}
    if processed_features.empty:
        return {"predicted_label": "Error: Processed features are empty.", "predicted_level": -1, "probabilities": {}}

    try:
        prediction_proba = full_model_pipeline.predict_proba(processed_features)[0]
        predicted_class = full_model_pipeline.predict(processed_features)[0]

        label_mapping = {
            0: "Low/No Burnout",
            1: "Mild Burnout",
            2: "Moderate Burnout",
            3: "High Burnout"
        }
        predicted_label = label_mapping.get(predicted_class, "Unknown Level")
        predicted_level = int(predicted_class)

        # Ensure class_labels are consistent with the model's output
        class_labels = full_model_pipeline.classes_
        probabilities = {label_mapping.get(cls, f"Class {cls}"): prob for cls, prob in zip(class_labels, prediction_proba)}

        return {
            "predicted_label": predicted_label,
            "predicted_level": predicted_level,
            "probabilities": probabilities
        }

    except Exception as e:
        return {"predicted_label": f"Error during model prediction: {e}", "predicted_level": -1, "probabilities": {}}