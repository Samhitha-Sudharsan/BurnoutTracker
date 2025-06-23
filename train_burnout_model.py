import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import randint
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

print("Loading data from final_combined_training_data.csv...")
try:
    df = pd.read_csv('final_combined_training_data.csv')
    print("Data loaded successfully.")
    print(f"Initial DataFrame shape: {df.shape}")
    print("Initial DataFrame columns:", df.columns.tolist())
except FileNotFoundError:
    print("Error: 'final_combined_training_data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

print("\nCleaning 'burnout' column...")
burnout_mapping = {
    'Not at all': 0, 'Mild tiredness': 1, 'Drained': 2, 'Full-on burnout': 3,
    '0.0': 0, '1.0': 1, '2.0': 2, '3.0': 3
}
df['burnout'] = df['burnout'].astype(str).map(burnout_mapping)
df['burnout'] = pd.to_numeric(df['burnout'], errors='coerce')
print(f"Burnout column unique values after initial cleaning: {df['burnout'].unique()}")

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    initial_rows_temp = len(df)
    df.dropna(subset=['date'], inplace=True)
    if len(df) < initial_rows_temp:
        print(f"Dropped {initial_rows_temp - len(df)} rows due to NaN in 'date' column for temporal features.")
    print("Converted 'date' column to datetime and handled NaNs.")
else:
    print("Warning: 'date' column not found in DataFrame. Cannot create temporal features.")

numerical_cols = ['age', 'mood', 'anxiety', 'energy', 'sleep']

categorical_cols = [
    'refreshed_after_sleep', 'steps_taken', 'water_intake',
    'caffeine_intake', 'work_hours',
    'outfit_type', 'music_volume', 'music_time', 'on_period_today',
    'cycle_phase', 'symptoms_experienced', 'music_genre'
]

user_profile_columns = [
    'occupation', 'alcohol_consumption', 'prior_burnout_anxiety',
    'living_situation', 'supportive_environment',
    'liking_study_work_environment', 'contraceptive_pill_use',
    'drug_use_smoking_habits'
]
for col in user_profile_columns:
    if col in df.columns and df[col].dtype == 'object':
        categorical_cols.append(col)

original_text_cols = ['journal_entry', 'major_event_log']

numerical_cols = [col for col in numerical_cols if col in df.columns]
categorical_cols = [col for col in categorical_cols if col in df.columns]
original_text_cols = [col for col in original_text_cols if col in df.columns]

print(f"\nIdentified Numerical Columns: {numerical_cols}")
print(f"Identified Categorical Columns: {categorical_cols}")
print(f"Original Text Columns for Concatenation: {original_text_cols}")

print("\nInitiating Temporal Feature Engineering...")
temporal_cols_to_process = ['mood', 'anxiety', 'energy', 'sleep']
temporal_cols_to_process = [col for col in temporal_cols_to_process if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

if 'user_id' in df.columns and 'date' in df.columns and temporal_cols_to_process:
    df.sort_values(by=['user_id', 'date'], inplace=True)
    print("DataFrame sorted for temporal feature calculation.")
    for col in temporal_cols_to_process:
        df[f'{col}_3day_avg'] = df.groupby('user_id')[col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{col}_3day_std'] = df.groupby('user_id')[col].rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)
        df[f'{col}_7day_avg'] = df.groupby('user_id')[col].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{col}_7day_std'] = df.groupby('user_id')[col].rolling(window=7, min_periods=1).std().reset_index(level=0, drop=True)
    print("Temporal feature engineering complete.")
else:
    print("Skipping temporal feature engineering: Missing 'user_id', 'date' or no relevant numerical columns for temporal processing.")

newly_created_temporal_features = []
for col in temporal_cols_to_process:
    newly_created_temporal_features.extend([
        f'{col}_3day_avg', f'{col}_3day_std',
        f'{col}_7day_avg', f'{col}_7day_std'
    ])
numerical_cols.extend([f for f in newly_created_temporal_features if f in df.columns])
numerical_cols = list(set(numerical_cols))

for col in original_text_cols:
    if col in df.columns:
        df[col] = df[col].fillna('')

if original_text_cols:
    print(f"\nConcatenating {original_text_cols} into a single 'combined_text' column.")
    df['combined_text'] = df[original_text_cols].agg(' '.join, axis=1)
    text_cols_for_vectorization = ['combined_text']
    print(f"'combined_text' column created with shape: {df['combined_text'].shape}")
else:
    text_cols_for_vectorization = []
    print("\nNo original text columns found to combine for NLP.")

TARGET_COLUMN = 'burnout'

initial_rows = len(df)
df.dropna(subset=[TARGET_COLUMN], inplace=True)
rows_after_nan_drop = len(df)
if initial_rows - rows_after_nan_drop > 0:
    print(f"\nDropped {initial_rows - rows_after_nan_drop} rows due to NaN in '{TARGET_COLUMN}' column after conversion.")
    print(f"Remaining rows: {rows_after_nan_drop}")

y = df[TARGET_COLUMN].astype(int)

columns_to_drop_from_X = [TARGET_COLUMN, 'date', 'user_id'] + original_text_cols
X = df.drop(columns=columns_to_drop_from_X, errors='ignore')

X = X.loc[y.index]

print(f"\nFeatures (X) shape after final selection: {X.shape}")
print(f"Target (y) shape after final selection: {y.shape}")

class BertEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        texts = X.iloc[:, 0].tolist() if isinstance(X, pd.DataFrame) else X.tolist()

        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings

    def get_feature_names_out(self, input_features=None):
        embedding_dim = 384
        return [f"bert_embed_{i}" for i in range(embedding_dim)]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown_Category')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

bert_text_transformer = Pipeline(steps=[
    ('bert_embedder', BertEmbeddingTransformer())
])

features_to_scale_actual = [f for f in numerical_cols if f in X.columns]
features_to_encode_actual = [f for f in categorical_cols if f in X.columns]
features_to_vectorize_actual = [f for f in text_cols_for_vectorization if f in X.columns]


print(f"\nFinal numerical features for scaling: {features_to_scale_actual}")
print(f"Final categorical features for encoding: {features_to_encode_actual}")
print(f"Final text features for BERT vectorization: {features_to_vectorize_actual}")


transformers_list = []
if features_to_scale_actual:
    transformers_list.append(('num', numerical_transformer, features_to_scale_actual))
if features_to_encode_actual:
    transformers_list.append(('cat', categorical_transformer, features_to_encode_actual))
if features_to_vectorize_actual:
    transformers_list.append(('text', bert_text_transformer, features_to_vectorize_actual))

if not transformers_list:
    print("Error: No features available for preprocessing. Exiting.")
    exit()

preprocessor = ColumnTransformer(
    transformers=transformers_list,
    remainder='drop'
)

print("\nApplying final preprocessing pipeline (including BERT embeddings for combined text data)...")
X_processed = preprocessor.fit_transform(X)

all_feature_names = []
if 'num' in preprocessor.named_transformers_ and features_to_scale_actual:
    all_feature_names.extend(preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out(features_to_scale_actual))
if 'cat' in preprocessor.named_transformers_ and features_to_encode_actual:
    all_feature_names.extend(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(features_to_encode_actual))
if 'text' in preprocessor.named_transformers_ and features_to_vectorize_actual:
    all_feature_names.extend(preprocessor.named_transformers_['text'].named_steps['bert_embedder'].get_feature_names_out(features_to_vectorize_actual))


X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)

print(f"\nProcessed features (X_processed_df) shape: {X_processed_df.shape}")
print("Processed DataFrame Info:")
X_processed_df.info()
print("\nFirst 5 rows of processed features (note BERT embeddings):")
print(X_processed_df.head())
print("\nFinal check for NaNs in X_processed_df (should be 0 for all columns):")
print(X_processed_df.isnull().sum().sum())


print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

print("\nInitiating RandomForestClassifier Hyperparameter Tuning with RandomizedSearchCV (including BERT NLP features)...")

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_features': ['sqrt', 'log2'],
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy']
}

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='f1_weighted'
)

random_search.fit(X_train, y_train)

print("\nRandomizedSearchCV complete.")
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation score (f1_weighted): {random_search.best_score_:.4f}")

best_model = random_search.best_estimator_

print("\nEvaluating the best model on the test set...")
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}")

print("\nClassification Report (Best Model from RandomizedSearchCV):")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix (Best Model from RandomizedSearchCV):")
print(confusion_matrix(y_test, y_pred))

MODEL_SAVE_PATH = 'burnout_prediction_random_forest_model_tuned_with_bert_nlp.pkl'
PREPROCESSOR_SAVE_PATH = 'data_preprocessor_for_random_forest_tuned_with_bert_nlp.pkl'

joblib.dump(best_model, MODEL_SAVE_PATH)
joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)

print(f"\nBest model saved to {MODEL_SAVE_PATH}")
print(f"Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")

print("\nML pipeline execution complete with RandomForestClassifier (tuned with RandomizedSearchCV), temporal features, and BERT NLP features.")
print(f"Target accuracy: 0.70. Achieved accuracy: {accuracy:.4f}")
print("Review the classification report above to assess performance across classes.")
