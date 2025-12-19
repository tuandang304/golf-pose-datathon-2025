
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
PROCESSED_FEATURES_DIR = "features"
VIDEO_METADATA_PATH = "video_metadata.csv"
OUTPUT_CSV_PATH = "features.csv"

# --- MEDIAPIPE LANDMARK INDICES ---
NOSE, MOUTH_L, MOUTH_R = 0, 9, 10
SHOULDER_L, SHOULDER_R = 11, 12
HIP_L, HIP_R = 23, 24
KNEE_L, KNEE_R = 25, 26
ANKLE_L, ANKLE_R = 27, 28
ELBOW_L, ELBOW_R = 13, 14
WRIST_L, WRIST_R = 15, 16

# --- FEATURE ENGINEERING CLASS ---

class FeatureEngineer:
    def __init__(self, metadata_path, features_dir):
        self.video_df = pd.read_csv(metadata_path)
        self.features_dir = features_dir
        self.phase_names = ['address', 'backswing', 'downswing', 'follow_through']
        self.feature_columns = []

    def calculate_angle(self, p1, p2, p3):
        """Calculates the angle (in degrees) between three points."""
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        return angle

    def get_static_features(self, pose):
        """Calculates a dictionary of static features from a single pose."""
        features = {}
        
        # Keypoints
        shoulder_l, shoulder_r = pose[SHOULDER_L], pose[SHOULDER_R]
        hip_l, hip_r = pose[HIP_L], pose[HIP_R]
        ankle_l, ankle_r = pose[ANKLE_L], pose[ANKLE_R]
        nose = pose[NOSE]

        # Spine and Pelvis centers
        spine_center = (shoulder_l + shoulder_r) / 2
        pelvis_center = (hip_l + hip_r) / 2

        # Spine Vertical Angle
        spine_vec = spine_center - pelvis_center
        vertical_vec = np.array([0, -1, 0]) # Assuming Y is up
        features['spine_vertical_angle'] = self.calculate_angle(pelvis_center + vertical_vec, pelvis_center, spine_center)

        # Stance Width / Shoulder Width Ratio
        stance_width = np.linalg.norm(ankle_l - ankle_r)
        shoulder_width = np.linalg.norm(shoulder_l - shoulder_r)
        if shoulder_width > 1e-6:
            features['stance_shoulder_ratio'] = stance_width / shoulder_width
        else:
            features['stance_shoulder_ratio'] = 0

        # Head-Spine Alignment (distance of nose from spine line)
        # Project nose onto the spine line
        ap = nose - pelvis_center
        ab = spine_center - pelvis_center
        projection = pelvis_center + np.dot(ap, ab) / np.dot(ab, ab) * ab
        features['head_spine_alignment'] = np.linalg.norm(nose - projection)

        return features

    def get_dynamic_features(self, pose_sequence, fps):
        """Calculates aggregated dynamic features from a pose sequence (phase)."""
        if len(pose_sequence) < 2:
            return {}
            
        features = {}
        
        # Joint Angles over time
        knee_l_angles = [self.calculate_angle(p[HIP_L], p[KNEE_L], p[ANKLE_L]) for p in pose_sequence]
        elbow_r_angles = [self.calculate_angle(p[SHOULDER_R], p[ELBOW_R], p[WRIST_R]) for p in pose_sequence]

        # Aggregations
        for name, data in [('knee_l_angle', knee_l_angles), ('elbow_r_angle', elbow_r_angles)]:
            features[f'{name}_mean'] = np.mean(data)
            features[f'{name}_std'] = np.std(data)
            features[f'{name}_max'] = np.max(data)
            features[f'{name}_range'] = np.max(data) - np.min(data)
        
        # Angular Velocity
        knee_l_velocity = np.abs(np.diff(knee_l_angles)) * fps
        features['knee_l_ang_vel_mean'] = np.mean(knee_l_velocity)
        features['knee_l_ang_vel_max'] = np.max(knee_l_velocity)

        return features

    def run(self):
        """Main execution method."""
        all_features = []

        for index, row in tqdm(self.video_df.iterrows(), total=self.video_df.shape[0]):
            video_path = row["video_path"]
            relative_path = os.path.relpath(video_path, start="CUSTOM_DATASET")
            base_name = os.path.basename(relative_path).replace(".mp4", "")
            
            video_features = {"video_path": video_path, "error_name": row["error_name"]}

            # --- Get Static Features ---
            # For ALL videos, we compute static features from the representative "address" pose.
            # For static videos, we load the single representative pose.
            # For dynamic videos, we use the "address" phase.
            static_pose_path = os.path.join(self.features_dir, "static", base_name + "_static.npy")
            address_pose_path = os.path.join(self.features_dir, "dynamic", base_name + "_address.npy")

            pose_for_static_features = None
            if os.path.exists(static_pose_path):
                pose_for_static_features = np.load(static_pose_path)
            elif os.path.exists(address_pose_path):
                address_poses = np.load(address_pose_path)
                if len(address_poses) > 0:
                    pose_for_static_features = address_poses[0] # First frame of address

            if pose_for_static_features is not None:
                video_features.update(self.get_static_features(pose_for_static_features))

            # --- Get Dynamic Features ---
            dynamic_video_features = {}
            for phase in self.phase_names:
                phase_path = os.path.join(self.features_dir, "dynamic", f"{base_name}_{phase}.npy")
                phase_features = {}
                if os.path.exists(phase_path):
                    pose_sequence = np.load(phase_path)
                    phase_features = self.get_dynamic_features(pose_sequence, row['fps'])
                
                # Add prefix for the phase
                for key, value in phase_features.items():
                    dynamic_video_features[f'{phase}_{key}'] = value
            
            video_features.update(dynamic_video_features)
            all_features.append(video_features)

        # --- Finalize DataFrame ---
        feature_df = pd.DataFrame(all_features).fillna(0)
        feature_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Feature matrix saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    engineer = FeatureEngineer(VIDEO_METADATA_PATH, PROCESSED_FEATURES_DIR)
    engineer.run()

