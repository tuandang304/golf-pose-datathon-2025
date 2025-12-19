
import pandas as pd

# --- CONFIGURATION ---
ERROR_VECTORS_PATH = "error_vectors.csv"
OUTPUT_PATH = "predictions.csv"

def get_score_band(error_count):
    """Maps the total number of detected errors to a score band."""
    if error_count >= 4:
        return "1-2"
    elif error_count == 3:
        return "2-4"
    elif error_count == 2:
        return "4-6"
    elif error_count == 1:
        return "6-8"
    else: # 0 errors
        return "8-10"

def apply_scoring_rules():
    """
    Applies rule-based scoring to the detected error vectors.
    """
    # Load error vectors
    try:
        error_vectors_df = pd.read_csv(ERROR_VECTORS_PATH)
    except FileNotFoundError:
        print(f"Error: Error vectors file not found at {ERROR_VECTORS_PATH}. Please run the previous step.")
        return

    # Identify error columns (all columns except the first two metadata columns)
    error_cols = error_vectors_df.columns[2:]

    # Calculate total errors
    error_vectors_df['total_errors'] = error_vectors_df[error_cols].sum(axis=1)

    # Apply rule to get score band
    error_vectors_df['predicted_band'] = error_vectors_df['total_errors'].apply(get_score_band)

    # Find the list of detected errors
    def get_detected_errors(row):
        return [col for col in error_cols if row[col] == 1]
        
    error_vectors_df['detected_errors'] = error_vectors_df.apply(get_detected_errors, axis=1)

    # Select and reorder columns for the final output file
    output_df = error_vectors_df[['video_path', 'error_name', 'detected_errors', 'total_errors', 'predicted_band']]
    
    # Save the predictions
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Scoring complete. Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    apply_scoring_rules()
