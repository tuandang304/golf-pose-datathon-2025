
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import os

# --- CONFIGURATION ---
PREDICTIONS_PATH = "predictions.csv"
GROUND_TRUTH_PATH = "validation_ground_truth.csv"
OUTPUT_DIR = "validation_results"
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

# --- MAPPINGS ---
BAND_TO_CENTER = {
    "1-2": 1.5, "2-4": 3.0, "4-6": 5.0, "6-8": 7.0, "8-10": 9.0
}
BAND_LABELS = ["1-2", "2-4", "4-6", "6-8", "8-10"]

def validate():
    """
    Validates the predictions against ground truth and reports metrics.
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    try:
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
        truth_df = pd.read_csv(GROUND_TRUTH_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e.filename}. Please ensure both prediction and ground truth files exist.")
        return

    # Merge predictions with ground truth
    validation_df = pd.merge(predictions_df, truth_df, on="video_path")

    if validation_df.empty:
        print("Error: No matching videos found between predictions and ground truth.")
        return

    y_true = validation_df['true_band']
    y_pred = validation_df['predicted_band']

    # --- 1. Band Accuracy ---
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Band Accuracy: {accuracy:.2%}")

    # --- 2. MAE of Band Centers ---
    y_true_centers = y_true.map(BAND_TO_CENTER)
    y_pred_centers = y_pred.map(BAND_TO_CENTER)
    mae = mean_absolute_error(y_true_centers, y_pred_centers)
    print(f"Mean Absolute Error (on band centers): {mae:.2f}")

    # --- 3. Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=BAND_LABELS)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=BAND_LABELS, yticklabels=BAND_LABELS)
    plt.title('Score Band Confusion Matrix')
    plt.xlabel('Predicted Band')
    plt.ylabel('True Band')
    
    # Save the plot
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix plot saved to: {CONFUSION_MATRIX_PATH}")
    
    # --- Store metrics in a text file ---
    with open(os.path.join(OUTPUT_DIR, "validation_summary.txt"), "w") as f:
        f.write(f"Validation Metrics Summary\n")
        f.write("="*30 + "\n")
        f.write(f"Band Accuracy: {accuracy:.2%}\n")
        f.write(f"Mean Absolute Error (on band centers): {mae:.2f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(pd.DataFrame(cm, index=BAND_LABELS, columns=BAND_LABELS).to_string())

if __name__ == "__main__":
    validate()
