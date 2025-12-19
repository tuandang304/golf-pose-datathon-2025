
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mediapipe as mp

# --- CONFIGURATION ---
OUTPUT_DIR = "visualizations"
FEATURES_CSV = "features.csv"
ERROR_VECTORS_CSV = "error_vectors.csv"
VIDEO_METADATA_CSV = "video_metadata.csv"
POSE_DATA_DIR = "pose_data"

# MediaPipe drawing utils and connections
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# --- Main Visualization Function ---
def generate_visualizations():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    try:
        features_df = pd.read_csv(FEATURES_CSV)
        error_vectors_df = pd.read_csv(ERROR_VECTORS_CSV)
        video_meta_df = pd.read_csv(VIDEO_METADATA_CSV)
    except FileNotFoundError as e:
        print(f"Error: Missing data file {e.filename}. Please run all previous steps.")
        return

    print("Generating visualizations...")
    
    # 1. Feature Distribution Plot
    plot_feature_distribution(features_df, 'backswing_knee_l_angle_mean', 'bad_iron_swings', 'good_side_iron')

    # 2. Error Activation Heatmap
    plot_error_heatmap(error_vectors_df)

    # 3. Pose Overlay
    # Let's compare a 'bad_far_from_ball' swing with a 'good_body_posture_and_ball_distance' swing
    # Note: This requires manually finding suitable video paths. I'll use placeholders from the generated CSVs.
    bad_video_path = video_meta_df[video_meta_df['error_name'] == 'bad_far_from_ball']['video_path'].iloc[0]
    good_video_df = video_meta_df[video_meta_df['swing_quality'] == 'Good Swings']
    if not good_video_df.empty:
        good_video_path = good_video_df['video_path'].iloc[0]
        plot_pose_overlay(good_video_path, bad_video_path, "address_pose_comparison.png")
    else:
        print("Warning: Could not find a 'Good Swing' video for pose overlay.")

    print(f"Visualizations saved in '{OUTPUT_DIR}' directory.")

# --- Plotting Functions ---

def plot_feature_distribution(features_df, feature, bad_error_name, good_error_name_approx):
    """Plots the distribution of a feature for good vs. bad swings."""
    plt.figure(figsize=(10, 6))
    
    # Filter for the specific error and any good swing
    bad_swings = features_df[features_df['error_name'] == bad_error_name]
    good_swings = features_df[features_df['error_name'].str.contains('good', na=False)]
    
    if bad_swings.empty or good_swings.empty:
        print(f"Warning: Not enough data to plot distribution for feature '{feature}'.")
        return

    sns.kdeplot(bad_swings[feature], label=f'Bad Swings ({bad_error_name})', fill=True)
    sns.kdeplot(good_swings[feature], label='Good Swings', fill=True)
    
    plt.title(f'Distribution of "{feature}"')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{feature}.png"))
    plt.close()

def plot_error_heatmap(error_vectors_df):
    """Plots a heatmap of detected errors across videos."""
    plt.figure(figsize=(15, 10))
    
    # Use video path basename for clarity
    error_vectors_df['video_name'] = error_vectors_df['video_path'].apply(lambda x: os.path.basename(x))
    heatmap_data = error_vectors_df.set_index('video_name').drop(['video_path', 'error_name', 'total_errors', 'predicted_band', 'detected_errors'], axis=1, errors='ignore')
    
    if heatmap_data.shape[1] == 0:
        print("Warning: No error columns found to generate heatmap.")
        return

    sns.heatmap(heatmap_data, cmap='YlGnBu', cbar=False)
    plt.title('Error Activation Heatmap')
    plt.xlabel('Error Type')
    plt.ylabel('Video')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_activation_heatmap.png"))
    plt.close()

def plot_pose_overlay(good_video_path, bad_video_path, output_filename):
    """Draws two representative poses on the same plot for comparison."""
    fig, ax = plt.subplots()

    # Get representative pose for both videos (assuming they are static 'address' poses)
    good_pose = get_rep_pose_from_video(good_video_path)
    bad_pose = get_rep_pose_from_video(bad_video_path)

    if good_pose is None or bad_pose is None:
        print("Warning: Could not load poses for overlay plot.")
        return

    # Draw poses
    draw_pose_skeleton(ax, good_pose, color='blue', label='Good Pose')
    draw_pose_skeleton(ax, bad_pose, color='red', label='Bad Pose')

    ax.set_title('Pose Overlay: Good vs. Bad Setup')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis() # Invert y-axis to match image coordinates
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()

# --- Drawing and Data Helper Functions ---

def get_rep_pose_from_video(video_path):
    """Helper to load the representative pose for a video."""
    relative_path = os.path.relpath(video_path, start="CUSTOM_DATASET")
    base_name = os.path.basename(relative_path).replace(".mp4", "")
    
    # Try to load a static rep pose first, then an address phase
    static_path = os.path.join("features", "static", f"{base_name}_static.npy")
    address_path = os.path.join("features", "dynamic", f"{base_name}_address.npy")

    if os.path.exists(static_path):
        return np.load(static_path)
    elif os.path.exists(address_path):
        address_poses = np.load(address_path)
        if len(address_poses) > 0:
            return address_poses[0] # Use first frame of address
    return None

def draw_pose_skeleton(ax, landmarks, color, label):
    """Draws a 2D pose skeleton on a matplotlib Axes object."""
    # landmarks are in (X, Y, Z), we use X and Y for a 2D projection
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    
    # Plot joints
    ax.scatter(x, y, color=color, s=10, label=label)

    # Plot bones
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], color=color, linewidth=1.5)


if __name__ == '__main__':
    generate_visualizations()

