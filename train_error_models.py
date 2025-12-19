
import os
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm

# --- CONFIGURATION ---
FEATURES_CSV_PATH = "features.csv"
ERROR_METADATA_PATH = "error_metadata.csv"
OUTPUT_DIR = "error_models"
MIN_SAMPLES_FOR_TRAINING = 5

def train_error_models():
    """
    Trains a One-Class SVM for each error type to learn its feature representation.
    """
    # Load data
    try:
        features_df = pd.read_csv(FEATURES_CSV_PATH)
        error_meta_df = pd.read_csv(ERROR_METADATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e.filename}. Please run previous steps.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get list of unique errors to model
    error_names = error_meta_df[error_meta_df['video_count'] >= MIN_SAMPLES_FOR_TRAINING]['error_name'].unique()
    
    if len(error_names) == 0:
        print(f"No errors with sufficient samples (>= {MIN_SAMPLES_FOR_TRAINING}) found to train models.")
        return

    # Identify feature columns (all numeric columns except metadata)
    feature_cols = features_df.select_dtypes(include=np.number).columns.tolist()
    
    print(f"Training One-Class SVM models for {len(error_names)} error types...")
    for error in tqdm(error_names):
        # Prepare training data: select only samples corresponding to the current error
        train_data = features_df[features_df['error_name'] == error][feature_cols]

        if len(train_data) < MIN_SAMPLES_FOR_TRAINING:
            print(f"Skipping '{error}': only {len(train_data)} samples, needs at least {MIN_SAMPLES_FOR_TRAINING}.")
            continue
            
        # Define the model pipeline
        # 1. StandardScaler: To normalize features, important for SVMs.
        # 2. OneClassSVM: To learn the boundary of the error class.
        # nu: an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', OneClassSVM(nu=0.1, kernel="rbf", gamma='auto'))
        ])

        # Train the model
        pipeline.fit(train_data)

        # Save the trained model
        output_path = os.path.join(OUTPUT_DIR, f"{error}.joblib")
        joblib.dump(pipeline, output_path)

    print("\nError model training complete.")
    print(f"Models saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    # Suppress numpy import warning if it occurs
    import numpy as np
    train_error_models()
