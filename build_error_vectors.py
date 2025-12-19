
import os
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
FEATURES_CSV_PATH = "features.csv"
MODELS_DIR = "error_models"
OUTPUT_CSV_PATH = "error_vectors.csv"

def build_error_vectors():
    """
    Uses the trained error models to generate an error vector for each video.
    """
    # Load feature data
    try:
        features_df = pd.read_csv(FEATURES_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Features file not found at {FEATURES_CSV_PATH}. Please run feature engineering first.")
        return

    # Find trained models
    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory not found at {MODELS_DIR}. Please run model training first.")
        return
        
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    if not model_files:
        print(f"Error: No trained models found in {MODELS_DIR}.")
        return

    error_names = [os.path.splitext(f)[0] for f in model_files]
    
    # Get feature columns
    feature_cols = features_df.select_dtypes(include=np.number).columns.tolist()
    X = features_df[feature_cols]

    # Prepare DataFrame to store results
    error_vectors_df = features_df[['video_path', 'error_name']].copy()

    print(f"Generating error vectors for {len(features_df)} videos using {len(error_names)} models...")

    # Predict with each model
    for error, model_file in tqdm(zip(error_names, model_files), total=len(model_files)):
        model_path = os.path.join(MODELS_DIR, model_file)
        model = joblib.load(model_path)
        
        # Predict returns 1 for inliers (error detected), -1 for outliers
        predictions = model.predict(X)
        
        # Map to 0 (no error) or 1 (error detected)
        error_detected = (predictions == 1).astype(int)
        error_vectors_df[error] = error_detected

    # Save the results
    error_vectors_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nError vectors saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    build_error_vectors()
