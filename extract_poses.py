
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_poses(video_metadata_path="video_metadata.csv", output_dir="pose_data"):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    # Load video metadata
    try:
        video_df = pd.read_csv(video_metadata_path)
    except FileNotFoundError:
        print(f"Error: {video_metadata_path} not found. Please run the metadata generation script first.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting poses for {len(video_df)} videos...")
    for index, row in tqdm(video_df.iterrows(), total=video_df.shape[0]):
        video_path = row["video_path"]
        
        # Define output path
        relative_path = os.path.relpath(video_path, start="CUSTOM_DATASET")
        output_path = os.path.join(output_dir, relative_path).replace(".mp4", ".npy")
        
        output_folder = os.path.dirname(output_path)
        os.makedirs(output_folder, exist_ok=True)

        if os.path.exists(output_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            continue

        landmarks_over_time = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and get pose landmarks
            results = pose.process(image_rgb)

            if results.pose_world_landmarks:
                # Use world landmarks for 3D coordinates in meters
                landmarks = results.pose_world_landmarks.landmark
                frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                landmarks_over_time.append(frame_landmarks)
        
        cap.release()

        if landmarks_over_time:
            # Shape: (T, num_joints, 3)
            pose_array = np.array(landmarks_over_time)
            np.save(output_path, pose_array)

    pose.close()
    print("Pose extraction complete.")
    print(f"Pose data saved in '{output_dir}' directory.")

if __name__ == "__main__":
    extract_poses()
