
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
POSE_DATA_DIR = "pose_data"
VIDEO_METADATA_PATH = "video_metadata.csv"
ERROR_METADATA_PATH = "error_metadata.csv"
OUTPUT_DIR = "features"
STATIC_DIR = os.path.join(OUTPUT_DIR, "static")
DYNAMIC_DIR = os.path.join(OUTPUT_DIR, "dynamic")

# MediaPipe landmark indices
WRIST_L, WRIST_R = 15, 16
ANKLE_L, ANKLE_R = 27, 28

# --- HELPER FUNCTIONS ---

def get_representative_pose(pose_data, window_size=5):
    """
    Finds the most stable frame in a pose sequence.
    Stability is defined as the minimum variance of all joint positions over a sliding window.
    """
    if len(pose_data) < window_size:
        # If video is too short, just take the middle frame
        return pose_data[len(pose_data) // 2]

    min_variance = float('inf')
    rep_frame_index = -1

    for i in range(len(pose_data) - window_size + 1):
        window = pose_data[i:i+window_size]
        # Variance across time dimension (axis 0) for all joints and coordinates
        variance = np.var(window, axis=0).sum()
        
        if variance < min_variance:
            min_variance = variance
            # Take the middle frame of the most stable window
            rep_frame_index = i + window_size // 2

    return pose_data[rep_frame_index]

def get_swing_phases(pose_data, fps):
    """
    Identifies swing phases based on hand speed and position.
    Returns a dictionary of pose sequences for each phase.
    """
    if len(pose_data) < 10: # Not enough frames for a dynamic swing
        return None

    # Calculate hand and feet center points over time
    hand_centers = np.mean(pose_data[:, [WRIST_L, WRIST_R], :], axis=1)
    feet_center = np.mean(pose_data[:, [ANKLE_L, ANKLE_R], :], axis=1)

    # Calculate hand speed (m/s)
    hand_velocities = np.linalg.norm(np.diff(hand_centers, axis=0), axis=1) * fps
    hand_speed = np.append(hand_velocities, hand_velocities[-1]) # Pad to match length

    # --- Phase Identification Heuristics ---
    
    # 1. Address/Setup phase: Starts at the beginning, ends when hands start moving up.
    # Find first significant hand movement.
    speed_threshold = 0.2 # m/s, tuning parameter
    start_of_backswing = np.where(hand_speed > speed_threshold)[0]
    if len(start_of_backswing) == 0:
        start_of_backswing = 5 # Default if no motion detected
    else:
        start_of_backswing = start_of_backswing[0]

    # 2. Top of Backswing: Point where hands are furthest from the feet.
    hand_to_feet_dist = np.linalg.norm(hand_centers - feet_center, axis=1)
    top_of_backswing = np.argmax(hand_to_feet_dist[:int(len(pose_data)*0.7)]) # Search in first 70%

    # 3. Impact: Point of maximum hand speed after the top of the backswing.
    if top_of_backswing + 5 < len(hand_speed):
        impact = top_of_backswing + np.argmax(hand_speed[top_of_backswing:])
    else:
        impact = len(hand_speed) - 1
        
    # Prevent index out of bounds
    start_of_backswing = min(start_of_backswing, top_of_backswing)
    top_of_backswing = max(start_of_backswing, top_of_backswing)
    impact = max(impact, top_of_backswing)

    phases = {
        "address": pose_data[0:start_of_backswing],
        "backswing": pose_data[start_of_backswing:top_of_backswing+1],
        "downswing": pose_data[top_of_backswing+1:impact+1],
        "follow_through": pose_data[impact+1:]
    }

    # Filter out empty phases
    return {name: data for name, data in phases.items() if len(data) > 0}


def process_swings():
    """
    Main function to process all poses and split them into static/dynamic features.
    """
    # Load metadata
    video_df = pd.read_csv(VIDEO_METADATA_PATH)
    error_df = pd.read_csv(ERROR_METADATA_PATH)
    error_type_map = pd.Series(error_df.type.values, index=error_df.error_name).to_dict()

    # Create output directories
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(DYNAMIC_DIR, exist_ok=True)

    print("Processing swings into static poses and dynamic phases...")
    for index, row in tqdm(video_df.iterrows(), total=video_df.shape[0]):
        video_path = row["video_path"]
        error_name = row["error_name"]
        
        # Determine output file name
        relative_path = os.path.relpath(video_path, start="CUSTOM_DATASET")
        base_output_path = os.path.join(OUTPUT_DIR, relative_path).replace(".mp4", "")
        
        # Load pose data
        pose_path = os.path.join(POSE_DATA_DIR, relative_path).replace(".mp4", ".npy")
        if not os.path.exists(pose_path):
            print(f"Warning: Pose file not found: {pose_path}")
            continue
        
        pose_data = np.load(pose_path)
        if pose_data.shape[0] == 0:
            print(f"Warning: Empty pose file: {pose_path}")
            continue

        # Get error type ('static' or 'dynamic')
        swing_type = error_type_map.get(error_name, 'dynamic') # Default to dynamic for 'none' or good swings

        # --- Process and Save ---
        if swing_type == "static":
            rep_pose = get_representative_pose(pose_data)
            output_path = os.path.join(STATIC_DIR, os.path.basename(base_output_path) + "_static.npy")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, rep_pose)
        else: # dynamic
            phases = get_swing_phases(pose_data, row['fps'])
            if phases:
                for phase_name, phase_data in phases.items():
                    output_path = os.path.join(DYNAMIC_DIR, f"{os.path.basename(base_output_path)}_{phase_name}.npy")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    np.save(output_path, phase_data)

    print("Swing processing complete.")
    print(f"Static representative poses saved in: {STATIC_DIR}")
    print(f"Dynamic swing phases saved in: {DYNAMIC_DIR}")


if __name__ == "__main__":
    process_swings()
